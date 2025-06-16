# === utils/io.py ===

import pickle
import time

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def log_message(message, logfile=None):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"{timestamp} - {message}"
    print(full_msg)
    if logfile:
        with open(logfile, 'a') as f:
            f.write(full_msg + "\n")
