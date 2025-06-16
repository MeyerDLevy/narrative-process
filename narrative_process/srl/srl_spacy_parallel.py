import os
import warnings
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend

from .srl_component import get_nlp, expand_np

def perform_srl_on_sentence(nlp_instance, sentence, post_id):
    doc = nlp_instance(sentence)
    rows = []
    for verb in [t for t in doc if t.pos_ == "VERB"]:
        arg0 = str(expand_np(verb._.srl_arg0)) if verb._.srl_arg0 and verb._.srl_arg0 != "[Implied observer]" else str(verb._.srl_arg0 or "")
        arg1 = str(expand_np(verb._.srl_arg1)) if verb._.srl_arg1 else ""
        arg2 = str(expand_np(verb._.srl_arg2)) if verb._.srl_arg2 else ""
        neg_flag = "1" if verb._.srl_negated else "0"
        description = f"[V: {verb.text}] [ARG0: {arg0}] [ARG1: {arg1}] [ARG2: {arg2}]"
        rows.append({
            "unresolvedsentence": str(sentence),
            "sentence": str(sentence),
            "post_id": str(post_id),
            "description": str(description),
            "verb": str(verb.text),
            "arg0": arg0,
            "arg1": arg1,
            "arg2": arg2,
            "argM-NEG": neg_flag
        })
    return rows

def process_batch(batch_df):
    warnings.filterwarnings("ignore")
    local_nlp = get_nlp()
    batch_results = []
    for idx, row in tqdm(batch_df.iterrows()):
        text = str(row["text"])
        post_id = row.get("post_id", idx)
        try:
            doc = local_nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            for sent in sentences:
                batch_results.extend(perform_srl_on_sentence(local_nlp, sent, post_id))
        except Exception as e:
            raise RuntimeError(f"Error processing row={idx}, text[:50]={text[:50]}: {str(e)}") from None
    return batch_results

def process_texts_parallel(df, output_file, num_workers=8, batch_size=1000):
    batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
    with parallel_backend("loky"):
        results = Parallel(n_jobs=num_workers)(
            delayed(process_batch)(batch) for batch in tqdm(batches, desc="Processing Batches", unit="batch")
        )
    all_rows = [row for sublist in results for row in sublist]
    pd.DataFrame(all_rows).to_csv(output_file, index=False)
