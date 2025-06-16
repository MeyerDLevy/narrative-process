import sys
import os
import pandas as pd
import pdb

# Add package path
sys.path.append("G:\\Other computers\\My Computer\\Dropbox\\social science\\narrative process\\package\\")

from narrative_process.main import run_pipeline

def main():
    scriptsdir = "G:\\Other computers\\My Computer\\Dropbox\\social science\\narrative process\\package\\scripts"
    input_csv = pd.read_csv(os.path.join(scriptsdir, "incels test.csv"))
    results = run_pipeline(
        input_csv[["text"]],
        working_dir="D:\\scratch\\temp_output",
        verbose=False
    )
    df = results["rels"]
    print("Top clustered terms:")
    print(df.head())



'''
main()
'''


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _, _, tb = sys.exc_info()
        pdb.post_mortem(tb)
