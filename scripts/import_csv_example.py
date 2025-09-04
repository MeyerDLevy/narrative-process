import os
import pandas as pd

from narrative_process.main import run_pipeline


def main():
    """Run the narrative process pipeline on an example CSV."""

    scripts_dir = os.path.dirname(__file__)
    csv_path = os.path.join(scripts_dir, "incels test.csv")

    # Load example data. The CSV must contain a column named ``text``.
    df = pd.read_csv(csv_path)

    # Run the pipeline and capture the resulting relation dataframe
    results = run_pipeline(
        df[["text"]],
        working_dir=os.path.join(scripts_dir, "temp_output"),
        verbose=False,
    )

    rels = results["rels"]
    print("Top relation rows:")
    print(rels.head())


if __name__ == "__main__":
    main()
