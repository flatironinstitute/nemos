import os
from pathlib import Path
import matplotlib.pyplot as plt

# Get the READTHEDOCS_OUTPUT environment variable
root = os.environ.get("READTHEDOCS_OUTPUT")

if root:
    # Define the paths
    root_path = Path(root)
    static_path = root_path / "html/_static"
    target_path = static_path / "thumbnails/how_to_guide"

    # Check if both directories exist
    if static_path.exists() and target_path.exists():
        print(f"Both {static_path} and {target_path} exist. Plotting...")

        # Generate a sample plot
        plt.figure()
        plt.plot([1, 2, 3], [4, 5, 6], label="Sample Plot")
        plt.legend()
        plt.title("Post-Build Plot")

        # Save the plot in the target directory
        plot_path = target_path / "post_build_plot.svg"
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

        print(f"\nContents of {target_path}:")
        for item in target_path.iterdir():
            if item.is_file():
                print(f"- File: {item.name}")
            elif item.is_dir():
                print(f"- Directory: {item.name}")
    else:
        print(f"Either {static_path} or {target_path} does not exist.")
else:
    print("READTHEDOCS_OUTPUT is not set. Skipping plot generation.")
