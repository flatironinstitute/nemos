import os
from pathlib import Path

# Get the READTHEDOCS_OUTPUT environment variable
rtd_output = os.environ.get("READTHEDOCS_OUTPUT")

if rtd_output:
    print(f"READTHEDOCS_OUTPUT is set: {rtd_output}")

    # Convert to a Path object for better handling
    output_path = Path(rtd_output)
    print(f"Path resolved as: {output_path}")

    # Example: Check if it exists
    if output_path.exists():
        print(f"The path exists: {output_path}")
    else:
        print(f"The path does not exist: {output_path}")

else:
    print("READTHEDOCS_OUTPUT is not set.")
