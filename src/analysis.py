import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "Ocean_Health_Index_2018_global_scores.csv"
)

COLUMN_NAMES = {
    "AO": "Artisanal Opportunities",
    "BD": "Biodiversity",
    "CP": "Coastal Protection",
    "CS": "Carbon Storage",
    "CW": "Clean Waters",
    "ECO": "Economies",
    "FIS": "Fisheries",
    "FP": "Food Provision",
    "HAB": "Habitat",
    "ICO": "Iconic Species",
    "LE": "Livelihoods & Economies",
    "LIV": "Livelihoods",
    "LSP": "Sense of Place",
    "MAR": "Mariculture",
    "NP": "Natural Products",
    "SP": "Species Protection",
    "SPP": "Species Preservation",
    "TR": "Tourism & Recreation",
    "Index_": "Overall Index",
    "trnd_sc": "Trend Score"
}


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def select_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    desired = list(COLUMN_NAMES.keys())
    cols = [c for c in desired if c in df.columns]
    return df[cols].copy()


def correlation_matrix(scores: pd.DataFrame) -> pd.DataFrame:
    return scores.corr(numeric_only=True)


def print_corr_with_trend(corr: pd.DataFrame) -> pd.Series:
    s = corr["trnd_sc"].sort_values(ascending=False)
    print("\n--- Correlatie met trendscore ---")
    for code, val in s.items():
        full_name = COLUMN_NAMES.get(code, code)
        print(f"{full_name:30s}: {val:.3f}")
    return s


def plot_heatmap(corr: pd.DataFrame) -> None:
    """Heatmap met volledige namen op assen."""
    fig, ax = plt.subplots(figsize=(13, 10))
    im = ax.imshow(corr.values, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)

    labels = [COLUMN_NAMES.get(c, c) for c in corr.columns]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    ax.set_title("Correlation between Ocean Health Index Components", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_top_factors(corr_with_trend: pd.Series, top_n: int = 5) -> None:
    """Staafdiagram met volledige namen."""
    s = corr_with_trend.drop(labels=["trnd_sc"]).head(top_n)
    labels = [COLUMN_NAMES.get(i, i) for i in s.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, s.values, color="teal")
    ax.set_title(f"Top {top_n} Factors Correlated with Trend", fontsize=14)
    ax.set_ylabel("Correlation with Trend Score")
    ax.set_xticklabels(labels, rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def main():
    df = load_data()
    scores = select_score_columns(df)
    corr = correlation_matrix(scores)
    corr_with_trend = print_corr_with_trend(corr)
    plot_heatmap(corr)
    plot_top_factors(corr_with_trend, top_n=5)


if __name__ == "__main__":
    main()
