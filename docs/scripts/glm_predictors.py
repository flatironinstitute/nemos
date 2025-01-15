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
        annot_kws={"fontsize": 22},
    )
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(True)
    # Show plot
    plt.tight_layout(pad=2.0)


def generate_lfp_like_signal(duration, sampling_rate, seed):
    """
    Generate a synthetic LFP-like signal.

    Parameters:
    duration (float): Duration of the signal in seconds.
    sampling_rate (int): Sampling rate in Hz.

    Returns:
    np.ndarray: LFP-like signal.
    """
    np.random.seed(seed)  # For reproducibility

    # Time array
    t = np.linspace(0, duration, int(duration * sampling_rate))

    # Generate power spectrum with realistic parameters
    freqs = np.fft.rfftfreq(len(t), 1 / sampling_rate)

    # Power-law decay
    alpha = 1.0  # Moderate decay exponent for LFP signals
    power_spectrum = 10**3 / (freqs + 1e-6) ** alpha

    # Add a realistic peak at ~5 Hz (e.g., theta band)
    peak_freq = 5  # Hz
    peak_width = 2  # Width of the peak in Hz
    power_spectrum += np.exp(-0.5 * ((freqs - peak_freq) / peak_width) ** 2) * 300

    # Smooth transition to higher frequencies
    high_freq_decay = 0.3  # Slower decay for higher frequencies
    power_spectrum *= np.exp(-high_freq_decay * (freqs / 100))

    # Randomize phase
    random_phases = np.exp(2j * np.pi * np.random.rand(len(freqs)))

    # Combine magnitude and phase into the frequency domain signal
    signal_freq_domain = np.sqrt(power_spectrum) * random_phases

    # Inverse FFT to generate the time-domain signal
    signal = np.fft.irfft(signal_freq_domain)

    # Normalize signal to have zero mean and unit variance
    signal = (signal - np.mean(signal)) / np.std(signal)

    return t, signal


def plot_custom_features():
    # Parameters
    duration = 2  # seconds
    sampling_rate = 1000  # Hz
    num_signals = 7  # Number of signals to generate

    t = None
    signals = []
    for i in range(num_signals):
        t, signal = generate_lfp_like_signal(duration, sampling_rate, seed=i)
        signals.append(signal + i * 5)  # Shift each signal on the y-axis

    # Plot stacked signals
    plt.figure(figsize=(12, 4))
    cmap = plt.cm.get_cmap("rainbow")
    colors = cmap(np.linspace(0, 1, num_signals))

    ax = plt.subplot(111)
    for i, (signal, color) in enumerate(zip(signals, colors)):
        plt.plot(t, signal, color=color)

    plt.title("Custom Features", fontsize=40)
    plt.legend(loc="upper right")
    plt.tight_layout()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_facecolor("none")
