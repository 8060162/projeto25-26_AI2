import os
import sys

# Ensure the project root is available in sys.path so imports like
# "from Chunking.pipeline import run_pipeline" work even when running
# "python src/main.py" from the project root.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Chunking.pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline()
