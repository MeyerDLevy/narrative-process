# === utils/io.py ===

import pickle
import time

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def iter_pickles(path_or_file):
    """Yield objects sequentially from a pickle file containing multiple dumps."""
    close_handle = False
    if isinstance(path_or_file, str):
        f = open(path_or_file, "rb")
        close_handle = True
    else:
        f = path_or_file
    try:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break
    finally:
        if close_handle:
            f.close()

def log_message(message, logfile=None):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"{timestamp} - {message}"
    print(full_msg)
    if logfile:
        with open(logfile, 'a') as f:
            f.write(full_msg + "\n")
