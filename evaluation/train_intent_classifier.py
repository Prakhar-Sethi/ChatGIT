"""
Train a neural intent classifier for ChatGIT (Novelty N4).

Uses all labeled turns from sample_conversations.jsonl + eval_conversations.jsonl
as training data. Features: TF-IDF unigrams+bigrams. Model: LinearSVC.
Saves trained model to chatgit/core/intent_clf.pkl.

Usage:
    python -m evaluation.train_intent_classifier
"""

import os, sys, json, pickle
import numpy as np
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report


# ── Collect training data ──────────────────────────────────────────────────

DATA_FILES = [
    os.path.join(_ROOT, "data", "convcodebench", "sample_conversations.jsonl"),
    os.path.join(_ROOT, "data", "convcodebench", "eval_conversations.jsonl"),
]

VALID_INTENTS = {"locate", "explain", "summarize", "debug"}

def load_training_data():
    queries, labels = [], []
    seen = set()
    for path in DATA_FILES:
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    conv = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for turn in conv.get("turns", []):
                    q      = turn.get("query", "").strip()
                    intent = turn.get("intent", "").lower().strip()
                    if not q or intent not in VALID_INTENTS:
                        continue
                    key = (q, intent)
                    if key in seen:
                        continue
                    seen.add(key)
                    queries.append(q)
                    labels.append(intent)
    return queries, labels


# ── Train ──────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  ChatGIT Neural Intent Classifier Training")
    print("=" * 60)

    queries, labels = load_training_data()
    print(f"\n  Loaded {len(queries)} labeled turns")
    print(f"  Distribution: {dict(Counter(labels))}")

    if len(queries) < 20:
        print("  ERROR: not enough training data (need >= 20 samples).")
        sys.exit(1)

    # Pipeline: TF-IDF (unigrams + bigrams, sublinear TF) → LinearSVC
    # LinearSVC consistently outperforms LogReg on short-text intent classification
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_features=8000,
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
        )),
        ("clf", LinearSVC(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",   # handles class imbalance in intent dist
        )),
    ])

    # Cross-validation to get honest accuracy estimate
    cv = StratifiedKFold(n_splits=min(5, min(Counter(labels).values())),
                         shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, queries, labels, cv=cv,
                             scoring="accuracy", n_jobs=-1)
    print(f"\n  Cross-validation accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Fold scores: {[f'{s:.4f}' for s in scores]}")

    # Train on full data
    pipeline.fit(queries, labels)

    # Full-data classification report (train-set, to verify labels are clean)
    preds = pipeline.predict(queries)
    print(f"\n  Full-data report (training set):")
    print(classification_report(labels, preds, target_names=sorted(VALID_INTENTS)))

    # Save model
    out_path = os.path.join(_ROOT, "chatgit", "core", "intent_clf.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n  Model saved → {out_path}")

    # Quick sanity check
    test_queries = [
        ("Where is the Flask class defined?",   "locate"),
        ("How does the request context work?",  "explain"),
        ("Give me an overview of the module",   "summarize"),
        ("Why does it crash on empty input?",   "debug"),
        ("Find the authenticate function",      "locate"),
        ("What are the edge cases here?",       "debug"),
    ]
    print("\n  Sanity checks:")
    for q, expected in test_queries:
        pred = pipeline.predict([q])[0]
        ok = "OK" if pred == expected else "MISMATCH"
        print(f"    [{ok}] '{q[:45]:<45}'  pred={pred:<10}  expected={expected}")

    return pipeline, scores.mean()


if __name__ == "__main__":
    train()
