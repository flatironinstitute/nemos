import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss_history(loss_history: list[tuple], ax=None):
    """Plot ``loss_history`` per batch, marking epoch boundaries."""
    if ax is None:
        fig, ax = plt.subplots()

    batch_end_losses = [(i, t) for i, t in enumerate(loss_history) if t[0] == "batch_end"]
    train_begin_losses = [(i, t) for i, t in enumerate(loss_history) if t[0] == "train_begin"]

    epoch_end_losses = []
    for i in range(len(batch_end_losses) - 1):
        curr_idx, curr_tup = batch_end_losses[i]
        next_idx, next_tup = batch_end_losses[i + 1]
        if curr_tup[1] is None or next_tup[1] is None:
            continue
        if next_tup[1] - curr_tup[1] == 1:
            epoch_end_losses.append((curr_idx, curr_tup))

    if batch_end_losses:
        ax.plot(
            [t[0] for t in batch_end_losses],
            [t[1][3] for t in batch_end_losses],
            label="Loss per batch",
            alpha=0.7,
        )
    if train_begin_losses:
        ax.scatter(
            [t[0] for t in train_begin_losses],
            [t[1][3] for t in train_begin_losses],
            label="Loss at training start",
        )
    if epoch_end_losses:
        ax.scatter(
            [t[0] for t in epoch_end_losses],
            [t[1][3] for t in epoch_end_losses],
            label="Loss at epoch boundaries",
            zorder=100,
        )

    ax.set_xlabel("Log index (~batch index)")
    ax.set_ylabel("Loss value")

    ax.legend()
    sns.despine(ax=ax)

    return ax