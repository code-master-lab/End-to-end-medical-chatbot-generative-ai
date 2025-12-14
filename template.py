# ---------------------------------------------------------
# IMPORT SECTION
# ---------------------------------------------------------

import os
# os module is used to interact with the operating system.
# Here it helps us:
# - create folders
# - check if files exist
# - check file sizes

from pathlib import Path
# Path helps us handle file paths in a clean and OS-independent way.
# Instead of treating paths as plain strings, Path makes them objects.

import logging
# logging is used to show meaningful messages while the script runs.
# It helps us understand what the script is doing step-by-step.


# ---------------------------------------------------------
# LOGGING CONFIGURATION
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s:'
)

# This sets up logging behavior:
# - level=logging.INFO → show informational messages
# - format → show timestamp + message
# This is useful for debugging and understanding execution flow.


# ---------------------------------------------------------
# LIST OF FILES TO CREATE
# ---------------------------------------------------------

list_of_files = [
    "src/__init__.py",      # Makes src a Python package
    "src/helper.py",        # Helper functions (embeddings, loaders, etc.)
    "src/prompt.py",        # Prompt templates for LLM
    ".env",                 # Environment variables (API keys)
    "setup.py",             # Packaging & installation config
    "app.py",               # Main application entry point
    "research/trials.ipynb",# Notebook for experiments
    "test.py"               # Testing file
]

# This list defines the COMPLETE project structure.
# Each path represents a file we want to exist.
# If folders don't exist, they will be created automatically.


# ---------------------------------------------------------
# CORE LOGIC: CREATE FOLDERS AND FILES
# ---------------------------------------------------------

for filepath in list_of_files:

    # Convert string path into Path object
    filepath = Path(filepath)

    # Split path into:
    # filedir  → directory path (e.g., src, research)
    # filename → actual file name (e.g., helper.py)
    filedir, filename = os.path.split(filepath)


    # -----------------------------------------------------
    # STEP 1: CREATE DIRECTORY (IF NEEDED)
    # -----------------------------------------------------

    if filedir != "":
        # os.makedirs creates directory recursively.
        # exist_ok=True means:
        # - If directory already exists → no error
        os.makedirs(filedir, exist_ok=True)

        logging.info(
            f"Creating directory; {filedir} for the file: {filename}"
        )


    # -----------------------------------------------------
    # STEP 2: CREATE FILE (IF NOT EXISTS OR EMPTY)
    # -----------------------------------------------------

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):

        # Open file in write mode ("w"):
        # - Creates file if it doesn't exist
        # - Clears file if it exists but empty
        with open(filepath, "w") as f:
            pass
            # pass means:
            # "Create the file but don't write anything inside it"

        logging.info(f"Creating empty file: {filepath}")


    # -----------------------------------------------------
    # STEP 3: FILE ALREADY EXISTS
    # -----------------------------------------------------

    else:
        logging.info(f"{filename} is already exists")
