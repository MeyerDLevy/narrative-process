# === main.py ===
import os
import time
import pandas as pd
import numpy as np
import tempfile
from tqdm import tqdm
from narrative_process.utils.io import save_pickle, load_pickle, log_message, iter_pickles
from narrative_process.embeddings import bert, arg_embeddings
from narrative_process.clustering import cosine_matrix, agglomerative, community
from narrative_process.srl.srl_spacy_parallel import process_texts_parallel



def run_pipeline(
    input_df,
    working_dir=None,
    temp_dir=None,
    verbose=False,
    save_output=None,
    unique_sents=False,
    return_locals=False,
):
    """Run the full narrative processing pipeline.

    Parameters
    ----------
    input_df : pandas.DataFrame
        DataFrame containing a column named ``text`` with sentences to process.
    working_dir : str, optional
        Directory to store intermediate artifacts.
    temp_dir : str, optional
        Base directory for temporary working directory when ``working_dir`` is
        not provided.
    verbose : bool, default False
        Enable verbose logging.
    save_output : str, optional
        Path to save the final CSV output.
    unique_sents : bool, default False
        If True, deduplicate sentences before processing.
    return_locals : bool, default False
        If True, return a copy of the function's local variables under the key
        ``"locals"`` in the result dictionary.
    """
    if working_dir is None:
        temp_dir_path = tempfile.TemporaryDirectory(dir=temp_dir)
        working_dir = temp_dir_path.name
    else:
        temp_dir = None

    os.makedirs(working_dir, exist_ok=True)
    def log(msg):
        if verbose:
            log_message(msg)

    try:
        # === Paths ===
        srl_out = os.path.join(working_dir, "srlout.csv")
        embeddings_path = os.path.join(working_dir, "embeds.pickle")
        results_path = os.path.join(working_dir, "results.pickle")
        names_path = os.path.join(working_dir, "names.pickle")
        mm_path = os.path.join(working_dir, "cosine_similarity.mmap")
        term_embed_mat_path = os.path.join(working_dir, "term_embeds_mat.pickle")
        verb_mm_path = os.path.join(working_dir, "verb_cosine_similarity.mmap")
        verb_embed_mat_path = os.path.join(working_dir, "verb_embeds_mat.pickle")
        verb_splitdf_path = os.path.join(working_dir, "verb_splitdf.pickle")
        grouped_data_path = os.path.join(working_dir, "grouped_data.pickle")
        rels_mmfile_path = os.path.join(working_dir, "rels_cosine_similarity.mmap")

        # === Constants ===
        term_internal_cluster_min_thresh = 0.85
        term_max_prop = 0.025
        rel_internal_cluster_min_thresh = 0.8
        rel_max_prop = 0.025

        start = time.time()

        # === SRL ===
        if not os.path.exists(srl_out):
            log(f"Starting SRL on {len(input_df)} sentences")
            process_texts_parallel(input_df, srl_out, num_workers=16, batch_size=1000)
        rels = pd.read_csv(srl_out, low_memory=False)
        log(f"SRL completed in {time.time() - start:.2f} sec — {len(rels)} rows")

        # === Filter & Dedup ===
        df = rels.dropna(subset=['sentence'])
        if unique_sents:
            df = df.drop_duplicates(subset="sentence", keep="first")
            log(f"Deduplicated to {len(df)} unique sentences")
        df = df[(df['arg0'].str.strip() != '') & (df['arg1'].str.strip() != '')].reset_index(drop=True)
        log(f"{len(df)} sentences with non-empty arg0/arg1")

        # === Argument Embeddings ===
        summary = arg_embeddings.generate_embeddings(df, bert, output=embeddings_path)
        edicts = list(iter_pickles(embeddings_path))
        edf = pd.DataFrame(edicts)
        log(f"Generated embeddings for {summary['count']}")

        # === Averaged Argument Embeddings ===
        concatenated_args = df[['arg0', 'arg1']].values.ravel('K')
        unique_values = pd.unique(concatenated_args[~pd.isnull(concatenated_args)])
        log(f"{len(unique_values)} unique arg0/arg1 values")

        batch_size = 2500
        batches = [unique_values[i:i + batch_size] for i in range(0, len(unique_values), batch_size)]
        results = [arg_embeddings.process_batch(batch, df, edf) for batch in tqdm(batches)]
        averaged_embeddings = {}
        for batch_result in results:
            averaged_embeddings.update(batch_result)
        log(f"Averaged embeddings for {len(averaged_embeddings)} terms")

        save_pickle(results, results_path)
        names = list(averaged_embeddings.keys())
        save_pickle(names, names_path)

        embeddings_matrix = np.array(list(averaged_embeddings.values()), dtype=np.float16)
        save_pickle(embeddings_matrix, term_embed_mat_path)
        log(f"Embedding matrix shape: {embeddings_matrix.shape}")

        # === Cosine Similarity ===
        cosine_matrix.embeddingsmatrix2cosinesimmat(embeddings_matrix, memmap_file=mm_path)
        mm = np.memmap(mm_path, dtype='float16', shape=(len(names), len(names)), mode='r+')
        log("Cosine similarity matrix created")

        # === Clustering (Terms) ===
        clusters_named = agglomerative.agglomerative_clustering(
            embeddings_matrix, names, mmfile=mm_path,
            threshold=term_internal_cluster_min_thresh, maxclusterprop=term_max_prop)
        log(f"{len(clusters_named)} term clusters")

        names2mminddict = pd.DataFrame({"names": names}).reset_index().set_index('names')['index'].to_dict()
        cluster_central = community.get_most_central_terms(clusters_named, names2mminddict, mm)

        # === Argument Cluster Roll-up ===
        data = [(key, text) for key, texts in clusters_named.items() for text in texts]
        splitclusterdf = pd.DataFrame(data, columns=['key', 'argtext'])
        splitclusterdf = pd.merge(splitclusterdf, cluster_central, how="left", on="key")
        arg2rollup = pd.Series(splitclusterdf['mostcentralterm'].values, index=splitclusterdf['argtext']).to_dict()
        log(f"Created arg2rollup mapping for {len(arg2rollup)} terms")

        # === Verb Clustering ===
        unique_verbs = df['verb'].dropna().unique()
        log(f"{len(unique_verbs)} unique verbs")
        verb_batches = [unique_verbs[i:i + batch_size] for i in range(0, len(unique_verbs), batch_size)]
        verb_results = [arg_embeddings.process_batch(batch, df, edf, columns=('verb',)) for batch in tqdm(verb_batches)]
        averaged_verb_embeddings = {}
        for batch_result in verb_results:
            averaged_verb_embeddings.update(batch_result)
        log(f"Averaged embeddings for {len(averaged_verb_embeddings)} verbs")

        verb_names = list(averaged_verb_embeddings.keys())
        verb_embeddings_matrix = np.array(list(averaged_verb_embeddings.values()), dtype=np.float16)
        save_pickle(verb_embeddings_matrix, verb_embed_mat_path)
        cosine_matrix.embeddingsmatrix2cosinesimmat(verb_embeddings_matrix, memmap_file=verb_mm_path)
        verb_mm = np.memmap(verb_mm_path, dtype='float16', shape=(len(verb_names), len(verb_names)), mode='r+')
        log("Verb cosine similarity matrix created")

        verb_clusters_named = agglomerative.agglomerative_clustering(
            verb_embeddings_matrix, verb_names, mmfile=verb_mm_path,
            threshold=term_internal_cluster_min_thresh, maxclusterprop=term_max_prop)
        log(f"{len(verb_clusters_named)} verb clusters")

        verb_names2mminddict = pd.DataFrame({"names": verb_names}).reset_index().set_index('names')['index'].to_dict()
        verb_cluster_central = community.get_most_central_terms(verb_clusters_named, verb_names2mminddict, verb_mm)

        verb_data = [(key, text) for key, texts in verb_clusters_named.items() for text in texts]
        verb_splitdf = pd.DataFrame(verb_data, columns=['key', 'verb'])
        verb_splitdf = pd.merge(verb_splitdf, verb_cluster_central, how="left", on="key")
        save_pickle(verb_splitdf, verb_splitdf_path)
        verb2rollup = pd.Series(verb_splitdf['mostcentralterm'].values, index=verb_splitdf['verb']).to_dict()
        log(f"Created verb2rollup mapping for {len(verb2rollup)} verbs")

        # === Sentence Roll-up ===
        dfsmall = df[["post_id", "sentence", "description", "arg0", "verb", "arg1", "argM-NEG"]].fillna("")
        dfsmall["arg0cluster"] = dfsmall["arg0"].apply(lambda x: arg2rollup.get(x, x))
        dfsmall["arg1cluster"] = dfsmall["arg1"].apply(lambda x: arg2rollup.get(x, x))
        dfsmall["verbcluster"] = dfsmall["verb"].apply(lambda x: verb2rollup.get(x, x))
        dfsmallnew = dfsmall[["arg0cluster", "verbcluster", "arg1cluster", "argM-NEG", "post_id", "sentence"]]

        # === Relation Grouping ===
        edf['avg_embedding'] = edf.apply(lambda row: np.mean(np.vstack(row.dropna()), axis=0) if len(row.dropna()) > 0 else np.nan, axis=1)
        df_with_embeddings = dfsmallnew.join(edf['avg_embedding'])

        df_with_embeddings = df_with_embeddings.dropna(subset = ["avg_embedding"])

        #np.where(edf.avg_embedding.apply(lambda x: isinstance(x, np.ndarray)) != True)
        #np.where(df_with_embeddings.avg_embedding.apply(lambda x: isinstance(x, np.ndarray)) != True)

        grouped_data = df_with_embeddings.groupby(['arg0cluster', 'verbcluster', 'arg1cluster', "argM-NEG"])
        group_embeds = grouped_data['avg_embedding'].apply(lambda x: (np.mean(np.vstack([e for e in x if isinstance(e, np.ndarray)]), axis=0), len(x)))
        group_embeds = group_embeds.apply(pd.Series)
        group_embeds.columns = ['avg_embedding', 'count']
        group_sentences = grouped_data['sentence'].apply(list)
        group_postids = grouped_data['post_id'].apply(list)
        grouped_embeddings = group_embeds.join(group_sentences).join(group_postids).reset_index()
        save_pickle(grouped_embeddings, grouped_data_path)
        log(f"{len(grouped_embeddings)} relation tuples")

        # === Final Clustering of Relations ===
        groupdropna = grouped_embeddings.dropna().reset_index(drop=True)
        relembedsmat = np.vstack(groupdropna["avg_embedding"])
        relnames = groupdropna[["arg0cluster", "verbcluster", "arg1cluster", "argM-NEG"]].apply(lambda row: " / ".join([str(t) for t in row]), axis=1)

        cosine_matrix.embeddingsmatrix2cosinesimmat(relembedsmat, memmap_file=rels_mmfile_path)
        relmm = np.memmap(rels_mmfile_path, dtype='float16', shape=(len(relnames), len(relnames)), mode='r+')
        log("Relation cosine similarity matrix created")

        relclusters_named = agglomerative.agglomerative_clustering(
            relembedsmat, relnames, mmfile=rels_mmfile_path,
            threshold=rel_internal_cluster_min_thresh, maxclusterprop=rel_max_prop)
        log(f"{len(relclusters_named)} relation clusters")

        relnames2mminddict = pd.DataFrame({"names": relnames}).reset_index().set_index('names')['index'].to_dict()
        relcluster_central = community.get_most_central_terms(relclusters_named, relnames2mminddict, relmm)

        # === Output ===
        reldata = [(key, text) for key, texts in relclusters_named.items() for text in texts]
        relsplitdf = pd.DataFrame(reldata, columns=['key', 'argtext'])
        relsplitdf = pd.merge(relsplitdf, relcluster_central, how="left", on="key")
        relsplitdf['frequency'] = relsplitdf['mostcentralterm'].map(relsplitdf['mostcentralterm'].value_counts())
        relsplitdf = relsplitdf.sort_values(by='frequency', ascending=False)

        # Final aggregation
        termroll = groupdropna[["arg0cluster", "verbcluster", "arg1cluster", "argM-NEG", "count"]]
        termrollnamesjoined = termroll[["arg0cluster", "verbcluster", "arg1cluster", "argM-NEG"]].apply(lambda row: " / ".join([str(t) for t in row]), axis=1)
        trdf = pd.DataFrame({"argtext": termrollnamesjoined, "count": termroll["count"]})
        trdfmerge = pd.merge(trdf, relsplitdf, how="left", on="argtext")
        trdfmerge["combcol"] = trdfmerge.apply(lambda row: row["mostcentralterm"] if not pd.isna(row["mostcentralterm"]) else row["argtext"], axis=1)

        embcol = relembedsmat.tolist()
        newdf = trdfmerge.drop(columns=["key", "mostcentralterm"])
        newdf["embed"] = embcol
        newdf["sentences"] = groupdropna["sentence"]
        newdf["post_ids"] = groupdropna["post_id"]
        newdf["argtext"] = newdf["argtext"].apply(lambda x: [x])

        def safe_embed_mean(x):
            try:
                arrs = [e for e in x if isinstance(e, np.ndarray)]
                return np.mean(np.vstack(arrs), axis=0) if arrs else np.nan
            except Exception as ex:
                import pdb; pdb.set_trace()
                raise


        aggdf = newdf.groupby('combcol').agg({
            'count': 'sum',
            'embed': safe_embed_mean,
            'sentences': lambda x: sum(x, []),
            'post_ids': lambda x: sum(x, []),
            'argtext': lambda x: sum(x, [])
        }).reset_index()

        aggdf[['arg0', 'verb', 'arg1', "argM-NEG"]] = aggdf['combcol'].str.split('/', expand=True, n=3)
        rollupprint = aggdf[["combcol", "arg0", "verb", "arg1", "argM-NEG", "count", "sentences", "post_ids", "argtext"]]
        rollupprint["unique_post_count"] = rollupprint.post_ids.apply(lambda x: len(set(x)))
        rollupprint["unique_sentence_count"] = rollupprint.sentences.apply(lambda x: len(set(x)))
        rollupprint = rollupprint.sort_values(by="unique_post_count", ascending=False)

        log(f"Final output: {len(rollupprint)} rollup rows")

        if save_output:
            rollupprint.to_csv(save_output, index=False)
            log(f"Saved output to {save_output}")

        del mm
        del relmm

        result = {
            "rels": rels,
            "term_clusters": clusters_named,
            "term_centroids": cluster_central,
            "relation_clusters": relclusters_named,
            "relation_centroids": relcluster_central,
            "rollup": rollupprint,
        }

        if return_locals:
            # capture locals after cleanup; avoid including the result dictionary itself
            debug_locals = locals().copy()
            debug_locals.pop("result", None)
            result["locals"] = debug_locals

        return result

    finally:
        pass
