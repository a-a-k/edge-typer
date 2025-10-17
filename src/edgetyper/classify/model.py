"""
ML fallback to resolve ambiguous edges.

We train a small logistic regression on high-confidence rule labels (weak supervision),
then predict labels for 'unknown' edges. Features include:
  - p_messaging, link_ratio, median_lag_ns (scaled), p_overlap, p_nonneg_lag.

If there are too few high-confidence edges to train, we return rule labels only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


_FEATURES = ["p_messaging", "link_ratio", "median_lag_ns", "p_overlap", "p_nonneg_lag"]


def _prepare_xy(feat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = feat[_FEATURES].to_numpy(dtype=float)
    y = (feat["rule_label"] == "async").astype(int).to_numpy()
    return X, y


def label_with_fallback(
    rules_df: pd.DataFrame,
    random_state: int = 0,
) -> pd.DataFrame:
    # Split high-confidence (train) vs unknown (predict)
    train_df = rules_df[rules_df["rule_conf"] == "high"].copy()
    pred_df = rules_df[rules_df["rule_label"] == "unknown"].copy()

    if len(train_df) >= 2 and train_df["rule_label"].nunique() == 2:
        scaler = StandardScaler()
        X_train, y_train = _prepare_xy(train_df)
        X_train[:, 2] = np.log1p(np.maximum(0.0, X_train[:, 2]))  # log1p(median_lag_ns⁺)
        Xs = scaler.fit_transform(X_train)

        clf = LogisticRegression(max_iter=500, random_state=random_state)
        clf.fit(Xs, y_train)

        # Predict unknowns
        if not pred_df.empty:
            X_pred, _ = _prepare_xy(pred_df)
            X_pred[:, 2] = np.log1p(np.maximum(0.0, X_pred[:, 2]))
            Xsp = scaler.transform(X_pred)
            prob_async = clf.predict_proba(Xsp)[:, 1]
            pred_df["pred_label"] = np.where(prob_async >= 0.5, "async", "sync")
            pred_df["pred_score"] = prob_async
    else:
        # Not enough training data → default to "sync" for unknowns (conservative)
        if not pred_df.empty:
            pred_df["pred_label"] = "sync"
            pred_df["pred_score"] = 0.0

    # Keep high-confidence rule labels as-is
    keep_df = rules_df[rules_df["rule_conf"] == "high"].copy()
    keep_df["pred_label"] = keep_df["rule_label"]
    keep_df["pred_score"] = np.where(keep_df["rule_label"] == "async", 1.0, 0.0)

    out = pd.concat([keep_df, pred_df], ignore_index=True)
    return out[["src_service", "dst_service", "pred_label", "pred_score"]]
