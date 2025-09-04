import os
import pandas as pd


def main():
    csv_path = os.path.join(os.path.dirname(__file__), "incels test.csv")
    df = pd.read_csv(csv_path)
    print(df.head())


if __name__ == "__main__":
    main()
