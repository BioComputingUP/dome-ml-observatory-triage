"""Curated seed terms for the keyword lexicon -- not derived from TF-IDF/KeyBERT extraction or
human review, but a deliberately chosen, small, reviewable batch covering well-known ML/AI
vocabulary and non-methods publication-type words that the corpus-driven extraction may not
surface (either because it's rare in this particular training corpus, e.g. current LLM/agentic
terminology against an older paper corpus, or because a human reviewer simply hasn't reached it
yet). Version-controlled as plain data here (not a CSV) so the actual term list is reviewable via
git diff like any other code change -- see `pipeline/steps.py::step_keywords_seed_additional_terms`
for how these get filtered against already-decided/already-extracted terms and written out as a
separate "added terms" tier, never merged directly into the human-curated original files.
"""

from __future__ import annotations

ADDED_POSITIVE_TERMS: list[dict[str, str]] = [
    # ML algorithms -- supervised
    {"term": "linear regression", "category": "ml_algorithm_supervised"},
    {"term": "naive bayes", "category": "ml_algorithm_supervised"},
    {"term": "lightgbm", "category": "ml_algorithm_supervised"},
    {"term": "k-nearest neighbors", "category": "ml_algorithm_supervised"},
    {"term": "knn", "category": "ml_algorithm_supervised"},
    {"term": "recurrent neural network", "category": "ml_algorithm_supervised"},
    {"term": "rnn", "category": "ml_algorithm_supervised"},
    {"term": "long short-term memory", "category": "ml_algorithm_supervised"},
    {"term": "lstm", "category": "ml_algorithm_supervised"},
    {"term": "transformer", "category": "ml_algorithm_supervised"},
    # ML algorithms -- unsupervised
    {"term": "k-means clustering", "category": "ml_algorithm_unsupervised"},
    {"term": "hierarchical clustering", "category": "ml_algorithm_unsupervised"},
    {"term": "dbscan", "category": "ml_algorithm_unsupervised"},
    {"term": "principal component analysis", "category": "ml_algorithm_unsupervised"},
    {"term": "pca", "category": "ml_algorithm_unsupervised"},
    {"term": "t-sne", "category": "ml_algorithm_unsupervised"},
    {"term": "umap", "category": "ml_algorithm_unsupervised"},
    {"term": "autoencoder", "category": "ml_algorithm_unsupervised"},
    {"term": "gaussian mixture model", "category": "ml_algorithm_unsupervised"},
    {"term": "self-organizing map", "category": "ml_algorithm_unsupervised"},
    # Generative model terms
    {"term": "generative adversarial network", "category": "generative_model"},
    {"term": "gan", "category": "generative_model"},
    {"term": "variational autoencoder", "category": "generative_model"},
    {"term": "vae", "category": "generative_model"},
    {"term": "diffusion model", "category": "generative_model"},
    {"term": "generative model", "category": "generative_model"},
    {"term": "generative ai", "category": "generative_model"},
    # Agentic / LLM terms
    {"term": "large language model", "category": "agentic_llm"},
    {"term": "llm", "category": "agentic_llm"},
    {"term": "ai agent", "category": "agentic_llm"},
    {"term": "agentic ai", "category": "agentic_llm"},
    {"term": "multi-agent system", "category": "agentic_llm"},
    {"term": "autonomous agent", "category": "agentic_llm"},
    {"term": "retrieval-augmented generation", "category": "agentic_llm"},
    {"term": "rag", "category": "agentic_llm"},
    {"term": "foundation model", "category": "agentic_llm"},
    {"term": "prompt engineering", "category": "agentic_llm"},
    {"term": "fine-tuning", "category": "agentic_llm"},
    # Flagship biodata-type terms (small set, EDAM-adjacent top data paradigms)
    {"term": "proteomics", "category": "biodata_flagship"},
    {"term": "genomics", "category": "biodata_flagship"},
    {"term": "transcriptomics", "category": "biodata_flagship"},
    {"term": "metabolomics", "category": "biodata_flagship"},
    {"term": "microbiome", "category": "biodata_flagship"},
    {"term": "metagenomics", "category": "biodata_flagship"},
    {"term": "single-cell", "category": "biodata_flagship"},
    {"term": "multi-omics", "category": "biodata_flagship"},
]

# Specific technical abbreviations/terms that should survive keywords/lexicon_cleanup.py's
# within-list subsumption rule even when they're a token inside a longer approved phrase (e.g.
# "svm" inside "support vector machine svm") -- unlike genuinely generic words ("model",
# "learning", "network"), these are specific enough on their own to carry real signal. Reviewed
# and confirmed explicitly by the user after inspecting the first suggest-final-lexicon run's
# cleanup log.
PROTECTED_UNIGRAMS: set[str] = {
    "svm",
    "cnn",
    "roc",
    "xgboost",
    "regression",
    "classifier",
    "classification",
    "autoencoder",
}

ADDED_NEGATIVE_TERMS: list[dict[str, str]] = [
    {"term": "perspective", "category": "non_methods_pubtype"},
    {"term": "review", "category": "non_methods_pubtype"},
    {"term": "commentary", "category": "non_methods_pubtype"},
    {"term": "editorial", "category": "non_methods_pubtype"},
    {"term": "opinion", "category": "non_methods_pubtype"},
    {"term": "viewpoint", "category": "non_methods_pubtype"},
    {"term": "case report", "category": "non_methods_pubtype"},
    {"term": "letter to the editor", "category": "non_methods_pubtype"},
    {"term": "narrative review", "category": "non_methods_pubtype"},
    {"term": "correspondence", "category": "non_methods_pubtype"},
    {"term": "news", "category": "non_methods_pubtype"},
    {"term": "systematic review", "category": "non_methods_pubtype"},
    {"term": "meta-analysis", "category": "non_methods_pubtype"},
    {"term": "survey", "category": "non_methods_pubtype"},
]
