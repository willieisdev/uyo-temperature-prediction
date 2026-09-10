"""
shap_explain.py
----------------
Model-agnostic SHAP explainability for the persisted best model (Ch. 3.6 /
model interpretability). Works whether the best model is the stacking
ensemble, XGBoost alone, Random Forest, or Linear Regression, by explaining
`model.predict` on the *scaled* feature space via SHAP's general Explainer
(uses TreeExplainer automatically under the hood for tree models, and a
model-agnostic Permutation explainer otherwise).

Outputs (saved to reports/figures/):
    shap_summary_beeswarm.png  - per-feature impact & direction across samples
    shap_summary_bar.png       - mean |SHAP value| per feature (importance)
    shap_waterfall_example.png - single-prediction breakdown
"""

import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from modeling import DATA_PATH, MODEL_DIR, chronological_split

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")


def load_test_split(artifact):
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).sort_index()
    train_df, test_df = chronological_split(df)
    feature_cols = artifact["feature_cols"]
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    return X_train, X_test


def explain(
    model_path: str = os.path.join(MODEL_DIR, "best_model.pkl"),
    background_size: int = 100,
    sample_size: int = 300,
    random_state: int = 42,
):
    os.makedirs(FIG_DIR, exist_ok=True)
    artifact = joblib.load(model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_cols = artifact["feature_cols"]

    X_train, X_test = load_test_split(artifact)

    # Background & explanation sets, scaled the same way the model was trained on.
    rng = np.random.RandomState(random_state)
    bg_idx = rng.choice(len(X_train), size=min(background_size, len(X_train)), replace=False)
    sample_idx = rng.choice(len(X_test), size=min(sample_size, len(X_test)), replace=False)

    background = pd.DataFrame(scaler.transform(X_train.iloc[bg_idx]), columns=feature_cols)
    sample = pd.DataFrame(scaler.transform(X_test.iloc[sample_idx]), columns=feature_cols)

    explainer = shap.Explainer(model.predict, background, feature_names=feature_cols)
    shap_values = explainer(sample)

    # --- Beeswarm summary (direction + magnitude) ---
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    beeswarm_path = os.path.join(FIG_DIR, "shap_summary_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()

    # --- Bar summary (global feature importance) ---
    plt.figure()
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.tight_layout()
    bar_path = os.path.join(FIG_DIR, "shap_summary_bar.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()

    # --- Waterfall for a single example prediction ---
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    waterfall_path = os.path.join(FIG_DIR, "shap_waterfall_example.png")
    plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs_shap = (
        pd.Series(np.abs(shap_values.values).mean(axis=0), index=feature_cols)
        .sort_values(ascending=False)
    )

    print(f"Explained model: {artifact['model_name']}")
    print("Mean |SHAP value| per feature:")
    print(mean_abs_shap)
    print(f"\nSaved: {beeswarm_path}")
    print(f"Saved: {bar_path}")
    print(f"Saved: {waterfall_path}")

    return shap_values, mean_abs_shap


if __name__ == "__main__":
    explain()
