# Narrative Process

This repository contains a Python pipeline for extracting and clustering narrative relations from text.

## How it works

1. **Semantic Role Labeling (SRL)** – Uses a custom spaCy component to identify verbs and their arguments (ARG0, ARG1, etc.) in each sentence.
2. **Embedding generation** – Computes BERT embeddings for the arguments with `transformers` and `torch`.
3. **Clustering** – Groups similar arguments and relations using cosine similarity, agglomerative clustering and Louvain community detection (`scikit-learn`, `networkx`, `python-louvain`).
4. **Roll‑up and output** – Aggregates clustered relations and produces summary tables that show the most central terms.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare a CSV file with a column named `text` containing sentences.
3. Run the pipeline on the included example CSV:
   ```bash
   python scripts/import_csv_example.py
   ```
   This script loads a sample dataset and processes it through the narrative pipeline, printing the first few relation rows.
