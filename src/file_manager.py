import os
import shutil
from pathlib import Path

UPLOAD_FOLDER = "data/uploaded_files"

def ensure_upload_folder():
    Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

def save_uploaded_file(uploaded_file):
    ensure_upload_folder()
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def get_uploaded_files():
    ensure_upload_folder()
    return [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]

def remove_file(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False

def clear_upload_folder():
    shutil.rmtree(UPLOAD_FOLDER)
    ensure_upload_folder()