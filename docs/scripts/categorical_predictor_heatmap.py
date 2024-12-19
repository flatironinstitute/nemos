import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_categorical_var_design_matrix():
    # Data
    data = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )

    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(
        data,
        annot=True,
        fmt=".0f",
        cmap="Pastel1_r",
        cbar=False,
        linewidths=0.5,
        linecolor="black",
    )

    # Customization
    plt.title("Categorical Variables", fontsize=20)
    plt.xlabel("Predictors", fontsize=16)
    plt.ylabel("Samples", fontsize=16)
    plt.xticks(
        ticks=np.arange(data.shape[1]) + 0.5,
        labels=["P1", "P2", "P3", "P4"],
        rotation=0,
    )
    plt.yticks(
        ticks=np.arange(data.shape[0]) + 0.5,
        labels=["S1", "S2", "S3", "S4", "S5", "S6"],
        rotation=0,
    )
    plt.tight_layout()
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(True)
    # Show plot
    plt.tight_layout(pad=2.0)
