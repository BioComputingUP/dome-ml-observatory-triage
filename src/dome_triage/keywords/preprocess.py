"""Text preprocessing shared by TF-IDF and KeyBERT extraction.

Strips the inline pseudo-HTML section tags confirmed present in EPMC abstracts (e.g.
"<h4>Background</h4>") before tokenizing -- left unstripped, these polluted the original TF-IDF
pass: categorized_terms.csv (MLit-Triage-Nextflow) contains "h4"/"background"/"http" among its
top terms. Stripping them here is a direct fix over the original, not just a port.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import re
from functools import partial
from typing import Callable, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s-]")

_lemmatizer = WordNetLemmatizer()
_NLTK_RESOURCES = {
    "stopwords": "corpora/stopwords",
    "punkt_tab": "tokenizers/punkt_tab",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}

# `ensure_nltk_data()` does a `nltk.data.find()` filesystem search per resource, and
# `stopwords.words()` re-reads+re-parses the corpus file -- both are safe to do once and reuse.
# Live-measured cost of NOT caching these: 744,647-document run at 17.7 docs/sec (11h38m), with
# ensure_nltk_data() alone responsible for ~95% of per-call time. Caching restores ~467 docs/sec
# (~26x) with identical output -- this was a real production incident, not a hypothetical.
_nltk_data_ready = False
_stopwords_cache: set[str] | None = None


def ensure_nltk_data() -> None:
    """Downloads required NLTK corpora if missing. Baked into the Docker image at build time
    (see docker/Dockerfile.cpu) so this is a no-op there; kept here as a safety net for local
    (non-Docker) development. Memoized -- see module docstring above for why."""
    global _nltk_data_ready
    if _nltk_data_ready:
        return
    for package, resource_path in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)
    _nltk_data_ready = True


def _base_stopwords() -> set[str]:
    global _stopwords_cache
    if _stopwords_cache is None:
        _stopwords_cache = set(stopwords.words("english"))
    return _stopwords_cache


def strip_html_tags(text: str) -> str:
    return _HTML_TAG_RE.sub(" ", text)


def clean_text(text: str, extra_stopwords: Optional[set[str]] = None) -> str:
    """Strip HTML tags, tokenize, lowercase, drop stopwords/short/non-alpha tokens, lemmatize."""
    if not text:
        return ""
    ensure_nltk_data()
    stop = _base_stopwords() | (extra_stopwords or set())

    text = strip_html_tags(text)
    text = _NON_ALPHA_RE.sub(" ", text)
    tokens = word_tokenize(text.lower())
    tokens = [
        _lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop and len(token) > 2
    ]
    return " ".join(tokens)


def _available_memory_gb() -> Optional[float]:
    """Reads `MemAvailable` from `/proc/meminfo` (Linux-only, no extra dependency). Returns None
    if unreadable (e.g. non-Linux dev environment) so callers can fall back gracefully."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)  # kB -> GB
    except OSError:
        pass
    return None


def _default_worker_count(per_worker_budget_gb: float = 1.0) -> int:
    """Picks a worker count that respects both CPU count and *actually available* memory --
    tuned live against a real incident, not guessed. A first version of this function used
    Python's `spawn` start method with an uncapped `cpu_count() - 2` worker count. Each spawned
    worker had to independently re-import this whole application (`dome-triage`'s console-script
    entrypoint pulls in `pipeline/steps.py`, which imports `keybert_extract.py`, which imports
    `torch` -- multi-hundred-MB-to-GB per process) instead of sharing the parent's already-loaded
    memory. On this 15GB, 16-core machine (also running several unrelated Docker containers) that
    produced 14 independent multi-GB processes at once; the kernel OOM-killer fired 3 times inside
    six minutes killing `dome-triage` (confirmed directly via `journalctl`, ~10-11GB RSS each), and
    the host eventually needed a hard reset. Fixed two ways: (1) switched to `fork`, so workers
    share the parent's already-loaded memory via copy-on-write instead of each re-importing torch
    from scratch; (2) worker count is now capped by *available* memory, not just CPU count, so a
    shared/already-busy machine gets fewer workers automatically rather than repeating the
    incident. `per_worker_budget_gb=1.0` is deliberately generous/conservative -- fork's
    copy-on-write sharing means real per-worker cost is normally well under this, but CPython's
    reference counting touches (and thus copies) pages just by using them, so it's not zero
    either; this errs toward "fewer workers than technically possible" over "risk another crash."
    """
    cpu_cap = max(1, (os.cpu_count() or 4) - 2)
    available = _available_memory_gb()
    if available is None:
        return cpu_cap  # can't see memory (e.g. non-Linux) -- fall back to the CPU-only cap
    memory_cap = max(1, int(available // per_worker_budget_gb))
    return max(1, min(cpu_cap, memory_cap))


def parallel_map(
    func: Callable[[str], object],
    items: list[str],
    n_workers: Optional[int] = None,
    desc: str = "processing",
    on_progress: Optional[Callable[[int, int], None]] = None,
):
    """General-purpose parallel `map(func, items)` for CPU-bound, per-item work over a big corpus
    -- the shared machinery behind `clean_text_batch()` below, generalized so a caller needing
    *more* than one operation per document (e.g. Bm25Scorer needing both `clean_text()` and
    lexicon-matching per document) can do it in a single pass/single progress bar instead of two
    separate ones. `func` must be a module-level function (or a `functools.partial` of one) --
    not a local closure/lambda -- multiprocessing pickles task payloads through its internal queue
    regardless of start method, and pickle can't reconstruct a local object.

    **A generator, not a list-returning function** -- live-measured reason: at 744,647 real
    documents, an earlier version collected results into one big list and returned it, which
    forced every caller to hold that whole list *plus* whatever structures it then built from it
    (e.g. splitting each result into two separate per-document lists) simultaneously -- multiple
    full-corpus-sized copies alive at once. That version OOM-killed its own container (confirmed
    via `docker inspect`'s `OOMKilled: true`) right after finishing the parallel stage, during the
    ordinary-looking list comprehensions that unpacked the results. Yielding lets the caller build
    only the final structures it actually needs, one result at a time, instead of materializing an
    intermediate copy of the whole corpus first.

    Uses `multiprocessing`, not threads (CPU-bound work would get no benefit from threads, stuck
    behind the GIL), with `fork` (Linux default) explicitly rather than `spawn` -- see
    `_default_worker_count()`'s docstring for the *other* real incident (an OOM-killer-triggered
    *host* crash from `spawn`'s per-worker re-import cost) that choice is a direct response to.
    Worker count defaults to `_default_worker_count()` (capped by *available* memory, not just CPU
    count) unless overridden. One `Pool` is created for the whole call (not per-chunk), so
    start-up cost is paid once. `initializer=ensure_nltk_data` warms each worker's NLTK cache once
    at pool start, not once per document, mirroring `clean_text()`'s own single-process cache.

    Order of `items` is preserved (`imap`, not `imap_unordered`). `on_progress(processed, total)`,
    if given, fires roughly every 5% of items processed (not every item -- too chatty at 700k+
    scale) so a caller can print status / write a checkpoint file without re-implementing the
    chunking itself."""
    if not items:
        return
    workers = n_workers or _default_worker_count()
    total = len(items)
    chunksize = max(1, total // (workers * 20))
    checkpoint_every = max(1, total // 20)

    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=ensure_nltk_data) as pool:
        with tqdm(total=total, desc=f"{desc} ({workers} workers)", unit="doc") as bar:
            for i, result in enumerate(pool.imap(func, items, chunksize=chunksize), start=1):
                yield result
                bar.update(1)
                if on_progress and (i % checkpoint_every == 0 or i == total):
                    on_progress(i, total)


def clean_text_batch(
    texts: list[str],
    extra_stopwords: Optional[set[str]] = None,
    n_workers: Optional[int] = None,
    desc: str = "cleaning",
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list[str]:
    """Parallel `clean_text()` over a whole corpus -- see `parallel_map()` for the mechanics.
    Materializes the generator into a list (unlike `parallel_map` itself) since existing callers
    expect a list back; prefer `parallel_map()` directly for very large corpora where holding the
    full result list isn't needed."""
    return list(
        parallel_map(
            partial(clean_text, extra_stopwords=extra_stopwords),
            texts,
            n_workers=n_workers,
            desc=desc,
            on_progress=on_progress,
        )
    )
