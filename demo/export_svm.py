"""Rebuild the two CPU classifiers from Huy's notebook and verify their portable scores."""
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

from streamlit_app import preprocess, predict_scores

ROOT = Path(__file__).resolve().parents[1]


def split(name):
    with (ROOT / "data" / "raw" / f"{name}.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [row["tweet"] for row in rows], [row["label"] for row in rows]


def export():
    train, y_train = split("train")
    test, y_test = split("test")
    cleaned = [preprocess(text) for text in train]
    words = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                            sublinear_tf=True, max_features=150000)
    w, c = words.fit_transform(cleaned), chars.fit_transform(cleaned)
    baseline = LinearSVC(C=3, random_state=42).fit(w, y_train)
    improved = LinearSVC(C=3, random_state=42).fit(hstack([w, c]), y_train)
    assert list(baseline.classes_) == list(improved.classes_) == ["fake", "real"]
    wb, wc = baseline.coef_[0], improved.coef_[0]
    words_data = {term: [words.idf_[i], wb[i], wc[i]] for term, i in words.vocabulary_.items()}
    chars_data = {term: [chars.idf_[i], wc[i + w.shape[1]]] for term, i in chars.vocabulary_.items()}
    model = dict(words=words_data, chars=chars_data,
                 intercepts=[baseline.intercept_[0], improved.intercept_[0]])
    samples = [preprocess(text) for text in test]
    wt, ct = words.transform(samples), chars.transform(samples)
    expected = np.column_stack([baseline.decision_function(wt),
                                improved.decision_function(hstack([wt, ct]))])
    actual = np.array([predict_scores(text, model) for text in test])
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    for scores in actual.T:
        pred = np.where(scores > 0, "real", "fake")
        print(f"n={len(pred)} accuracy={accuracy_score(y_test, pred):.4f} "
              f"F1 macro={f1_score(y_test, pred, average='macro'):.4f}")
    dest = Path(__file__).with_name("svm.json.gz")
    payload = json.dumps(model, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    dest.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))


if __name__ == "__main__":
    export()
