from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import seaborn as sns

def highlight_max_cell(cvdf_wide, ax):
    max_col = cvdf_wide.max().idxmax()
    max_col_index = cvdf_wide.columns.get_loc(max_col)
    max_row = cvdf_wide[max_col].idxmax()
    max_row_index = cvdf_wide.index.get_loc(max_row)

    ax.add_patch(
        Rectangle(
            (max_col_index, max_row_index), 1, 1, fill=False, lw=3, color="skyblue"
        )
    )


def plot_heatmap_cv_results(cvdf_wide, label=None):
    plt.figure()
    ax = sns.heatmap(
        cvdf_wide,
        annot=True,
        square=True,
        linecolor="white",
        linewidth=0.5,
    )

    # Labeling the colorbar
    colorbar = ax.collections[0].colorbar
    if not label:
        colorbar.set_label('log-likelihood')
    else:
        colorbar.set_label(label)

    ax.set_xlabel("ridge regularization strength")
    ax.set_ylabel("number of basis functions")

    highlight_max_cell(cvdf_wide, ax)