import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer
from narrative_process.utils.io import save_pickle, load_pickle

from narrative_process.config import EMBEDDING_MODEL_NAME

# Use same model name as in bert.py
tokenizer = BertTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

def find_phrase_in_tokens(phrase, sentence_tokens, tokenizer):
    phrase_tokens = tokenizer.tokenize(str(phrase))
    for i in range(len(sentence_tokens) - len(phrase_tokens) + 1):
        if sentence_tokens[i:i + len(phrase_tokens)] == phrase_tokens:
            return i, i + len(phrase_tokens)
    return -1, -1

def generate_embeddings(df, bertembeds_module, output=None):
    """Generate embeddings for arguments in ``df``.

    If ``output`` is provided, each embedding dictionary is pickled to the
    given file path or file handle as it is computed. When writing to disk the
    function returns a summary dictionary containing the output path (when
    applicable) and the number of records written. If ``output`` is ``None`` the
    list of embedding dictionaries is returned as before.
    """

    close_handle = False
    handle = None
    summary = {"count": 0}

    if output is not None:
        if isinstance(output, str):
            handle = open(output, "wb")
            summary["path"] = output
            close_handle = True
        else:
            handle = output
            summary["path"] = getattr(output, "name", None)

    edicts = []

    for row in tqdm(df.itertuples(index=False), total=len(df)):
        row_dict = row._asdict()
        s = str(row_dict.get("sentence", ""))
        if len(tokenizer.tokenize(s)) > 500:
            embeddict = {"fullavg": np.nan}
        else:
            tokens, embeds = bertembeds_module.generate_bert_embeddings(s)
            args = {k: v for k, v in row_dict.items() if k not in ("sentence", "post_id", "description") and pd.notna(v)}
            embeddict = {}
            for argkey, argphrase in args.items():
                start, end = find_phrase_in_tokens(argphrase, tokens, tokenizer)
                if ((start != -1) or (end != -1)) and len(range(start, end)) > 0:
                    estack = [embeds[i] for i in range(start, end)]
                    avg = np.mean(estack, axis=0)
                    embeddict[argkey] = avg
            embeddict["fullavg"] = np.mean(list(embeddict.values()), axis=0) if embeddict else np.nan

        if handle is not None:
            pickle.dump(embeddict, handle)
            summary["count"] += 1
        else:
            edicts.append(embeddict)

    if handle is not None and close_handle:
        handle.close()

    return summary if handle is not None else edicts

def process(value, df, edf):
    indices_arg0 = df[df['arg0'] == value].index
    indices_arg1 = df[df['arg1'] == value].index
    embeddings_arg0 = edf["arg0"].iloc[indices_arg0].tolist()
    embeddings_arg1 = edf["arg1"].iloc[indices_arg1].tolist()
    all_embeddings = embeddings_arg0 + embeddings_arg1
    filtered_embeddings = [e for e in all_embeddings if not np.isnan(e).any()]
    if filtered_embeddings:
        return np.mean(filtered_embeddings, axis=0)
    return np.nan

def process_batch(values, df, edf, columns=("arg0", "arg1")):
    """Average embeddings for unique values across specified columns."""
    batch_result = {}
    for value in values:
        embeddings = []
        for col in columns:
            indices = df[df[col] == value].index
            embeddings.extend(edf[col].iloc[indices].tolist())
        filtered = [e for e in embeddings if isinstance(e, np.ndarray) and not np.isnan(e).any()]
        if filtered:
            batch_result[value] = np.mean(filtered, axis=0)
    return batch_result
