import os
from pathlib import Path
import logging
 
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')
 
 
list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "app.py",
    "frontend_part/chat.html",
    "frontend_part/style.css",
    "data/.gitkeep",
    "data2/.gitkeep",
    "research/trials.ipynb",
    "research/ingest_3rd_edition.ipynb",
]
 
 
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
 
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")
 
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
 
    else:
        logging.info(f"{filename} already exists and has content — skipped.")