#!/home2/s439820/anaconda3/envs/amr_supp/bin/python
"""AMR-family comprehensive sweep across all 10 cohorts × multiple AMR checkpoints.

Background (per user request):
  The user believes the AMR architecture (drug_transformer_AMR.py, 16 MB
  checkpoints) was the actual production training pipeline for the manuscript's
  reported per-cohort accuracies, and asks for testing all related saved weights.

Pipeline:
  - drug_transformer_AMR (15.3 MB ckpt family, 7-input signature with mask)
  - Same preprocessing as Micro_bio_train_test_AMR.py: 16-vocab StringLookup
    capped at num_tokens=8 (effectively F,S,N,O,I,L,B,C one-hot, rest -> zero)
  - test_dataset_merged.pkl filtered by __source__ for each cohort

Checkpoint groups tested:
  Group A — POOLED (test on all 10 cohorts):
    - ten_datasets_10_07_25_ACC0.848.h5 (highest acc named)
    - ten_datasets_10_07_25_ACC0.827.h5
    - ten_datasets_10_07_25.h5
    - ten_datasets_11_11_25.h5
    - Ten_datasets_1114.h5

  Group B — PER-COHORT NAMED AMR (test only on their named cohort):
    - antibiogram_1002_att2_edge6_pos2_new_train_15.h5 (latest antibiogram AMR)
    - antibiogram_1001_att2_edge6_pos2_new_train_10.h5
    - ARIsolateBank_0918.h5 (latest ARIsolateBank)
    - ARIsolateBank_0912.h5 (registered in PROVENANCE.md)
    - PATRIC_0913.h5

Output:
  - per-row CSV per (ckpt, cohort) pair
  - master metrics CSV (long form): one row per (ckpt, cohort)
"""
import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, '/home2/s439820/jupyter_notebooks/Drug_response-main')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
tf.keras.utils.set_random_seed(812)

from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
                              f1_score, matthews_corrcoef, confusion_matrix)
from utils.smile_rel_dist_interpreter import (
    generate_rel_dist_matrix, generate_interpret_smile, get_drug_edge_type,
)
import drug_transformer_AMR as mod

# ----------------------- constants -----------------------
DIM = 1962
MAX_ATOM = 85
BATCH_SIZE = 256
SAVED = '/home2/s439820/jupyter_notebooks/AMR_data/saved_model_0804'
OUT_DIR = '/home2/s439820/jupyter_notebooks/supplementary/intermediate'

# AMR vocab (16-element, capped at first 8 via num_tokens=8) — matches Micro_bio_train_test_AMR.py
VOCABULARY = ['F', 'S', 'N', 'O', 'I', 'L', 'B', 'C', 'A', 's', 'a', 'o', 'H', 'M', 'e', 'P']

# Cohort source markers (from test_dataset_merged.pkl)
COHORT_SOURCES = {
    'antibiogram':   'test_dataset2.pkl',
    'ARIsolateBank': 'test_dataset3.pkl',
    'AstraZeneca':   'test_dataset4.pkl',
    'CF':            'test_dataset5.pkl',
    'Chile':         'test_dataset6.pkl',
    'German':        'test_dataset7.pkl',
    'PATRIC':        'test_dataset8.pkl',
    'Rabin':         'test_dataset9.pkl',
    'Shelburne':     'test_dataset.pkl',
    'TIDB':          'test_dataset10.pkl',
}

# Sweep matrix: list of (checkpoint_name, list_of_cohorts_to_test)
ALL_COHORTS = list(COHORT_SOURCES.keys())
SWEEPS = [
    # ---------- Group A: pooled checkpoints — test on every cohort ----------
    ('ten_datasets_10_07_25_ACC0.848.h5', ALL_COHORTS),
    ('ten_datasets_10_07_25_ACC0.827.h5', ALL_COHORTS),
    ('ten_datasets_10_07_25.h5',          ALL_COHORTS),
    ('ten_datasets_11_11_25.h5',          ALL_COHORTS),
    ('Ten_datasets_1114.h5',              ALL_COHORTS),
    # ---------- Group B: per-cohort AMR — only their own cohort -----------
    ('antibiogram_1002_att2_edge6_pos2_new_train_15.h5', ['antibiogram']),
    ('antibiogram_1001_att2_edge6_pos2_new_train_10.h5', ['antibiogram']),
    ('ARIsolateBank_0918.h5', ['ARIsolateBank']),
    ('ARIsolateBank_0912.h5', ['ARIsolateBank']),
    ('PATRIC_0913.h5',        ['PATRIC']),
]

# ----------------------- positional encoding & helpers -----------------------
P = np.zeros((1, 100, 60))
XX = np.arange(100, dtype=np.float32).reshape(-1, 1) / np.power(1000, np.arange(0, 60, 2, dtype=np.float32) / 60)
P[:, :, 0::2] = np.sin(XX)
P[:, :, 1::2] = np.cos(XX)
P = tf.cast(tf.math.l2_normalize(P[:, :100, :], axis=-1), dtype=tf.float32)

edge_type_dict = np.zeros((5, 5))
for i in range(5):
    edge_type_dict[i, i] = 1
edge_type_dict = tf.cast(edge_type_dict, dtype=tf.float32)

gene_name_l_one_hot = [[1 if i == j else 0 for j in range(DIM)] for i in range(DIM)]

string_lookup = tf.keras.layers.StringLookup(vocabulary=VOCABULARY)
layer_one_hot = tf.keras.layers.CategoryEncoding(num_tokens=8, output_mode='one_hot')


# ----------------------- preprocessing (AMR-style: mask, not species) ----
def preprocess_chunk(rows):
    drug_rel_position_chunk, drug_smile_length_update, edge_type_matrix_chunk = [], [], []
    atom_chunk, ge_chunk, mu_chunk, label_chunk = [], [], [], []
    for r in rows:
        smiles = r['drug_smiles']
        ge_chunk.append(r['gene_expression'])
        mu_chunk.append(r['if_mutation'])
        label_chunk.append(int(r['label']))

        rel_distance_ = generate_rel_dist_matrix(smiles); s = rel_distance_.shape[0]
        rp = tf.cast(tf.gather(P[0], tf.cast(rel_distance_, tf.int32), axis=0), tf.float32)
        rp = tf.concat((rp, tf.zeros((MAX_ATOM - s, s, 60), dtype=tf.float32)), 0)
        rp = tf.concat((rp, tf.zeros((MAX_ATOM, MAX_ATOM - s, 60), dtype=tf.float32)), 1)
        drug_rel_position_chunk.append(rp)

        em = get_drug_edge_type(smiles); se = em.shape[0]
        em = tf.gather(edge_type_dict, tf.cast(em, tf.int16), axis=0)
        em = tf.concat((em, tf.zeros((MAX_ATOM - se, se, 5))), 0)
        em = tf.concat((em, tf.zeros((MAX_ATOM, MAX_ATOM - se, 5))), 1)
        edge_type_matrix_chunk.append(em)

        interpret_smile_ = generate_interpret_smile(smiles)
        atom_names = tf.constant(list(interpret_smile_[0]))
        atom_idx = string_lookup(atom_names) - 1
        atom_oh = layer_one_hot(atom_idx); sa = atom_oh.shape[0]
        atom_oh = tf.concat((atom_oh, tf.zeros((MAX_ATOM - sa, 8))), 0)
        atom_chunk.append(atom_oh)
        drug_smile_length_update.append(sa)

    drug_smile_length_update = np.array(drug_smile_length_update, dtype=np.int32)

    # Build mask of shape (B, MAX_ATOM, 1) per Micro_bio_train_test_AMR.py
    batch_shape = len(rows)
    mask = tf.range(start=0, limit=MAX_ATOM, dtype=tf.float32)
    mask = tf.broadcast_to(tf.expand_dims(mask, axis=0), shape=[batch_shape, MAX_ATOM])
    mask = tf.reshape(mask, shape=(batch_shape * MAX_ATOM,))
    mask = mask < tf.cast(tf.repeat(drug_smile_length_update, repeats=MAX_ATOM), tf.float32)
    mask = tf.where(mask, 1, 0)
    mask = tf.reshape(mask, shape=(batch_shape, MAX_ATOM))
    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.cast(mask, tf.float32)

    return (
        tf.stack(atom_chunk),
        tf.stack([tf.convert_to_tensor(np.array(x), dtype=tf.int32) for x in ge_chunk]),
        drug_smile_length_update,
        tf.stack(drug_rel_position_chunk),
        tf.stack(edge_type_matrix_chunk),
        tf.stack([tf.convert_to_tensor(np.array(x), dtype=tf.int32) for x in mu_chunk]),
        mask,
        np.array(label_chunk, dtype=np.int32),
    )


# ----------------------- load merged test data once --------------------
print('loading test_dataset_merged.pkl ...')
merged = pd.read_pickle('/home2/s439820/jupyter_notebooks/AMR_data/test_dataset_merged.pkl')
cohort_dfs = {}
for cohort, src in COHORT_SOURCES.items():
    df = merged[merged['__source__'] == src].reset_index(drop=True)
    cohort_dfs[cohort] = df
    print(f'  {cohort:<14s} {src:<25s} n={len(df)}')

# ----------------------- build model once, reload weights for each ckpt -
print('\nbuilding drug_transformer_AMR architecture ...')
t0 = time.time()
k = mod.drug_transformer_(gene_name_l_one_hot)
model = k.model_construction_midi(if_mutation=True)
print(f'  built in {time.time()-t0:.1f}s; params={model.count_params()} '
      f'weights={len(model.weights)}')


def run_one_cohort_for_current_weights(cohort, ckpt_label):
    """Run inference on a single cohort assuming model already has weights loaded."""
    df = cohort_dfs[cohort]
    n = len(df)
    preds_all, labels_all, drugs_all, species_all = [], [], [], []
    t = time.time()
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        rows = [df.iloc[i] for i in range(start, end)]
        atom_oh, ge, drug_len, rp, em, mu, mask, lbl = preprocess_chunk(rows)
        model_inputs = (atom_oh, ge, tf.constant(drug_len, dtype=tf.int32), rp, em, mu, mask)
        out = model(model_inputs)[:, 0].numpy()
        preds_all.extend(out.tolist())
        labels_all.extend(lbl.tolist())
        drugs_all.extend([r['drug_name'] for r in rows])
        species_all.extend([r['species_name'] for r in rows])
    elapsed = time.time() - t

    preds = np.asarray(preds_all)
    labels = np.asarray(labels_all)
    binary = (preds >= 0.5).astype(int)
    try:
        auroc = float(roc_auc_score(labels, preds))
    except ValueError:
        auroc = float('nan')
    try:
        auprc = float(average_precision_score(labels, preds))
    except ValueError:
        auprc = float('nan')
    acc = float(accuracy_score(labels, binary))
    f1 = float(f1_score(labels, binary, zero_division=0))
    mcc = float(matthews_corrcoef(labels, binary)) if len(set(labels.tolist())) > 1 else 0.0
    cm = confusion_matrix(labels, binary, labels=[0, 1]).tolist()

    metrics = {
        'checkpoint': ckpt_label,
        'cohort': cohort,
        'n': int(n),
        'pos_rate_truth': float(labels.mean()),
        'pred_rate_at_0.5': float(binary.mean()),
        'pred_prob_mean': float(preds.mean()),
        'pred_prob_min': float(preds.min()),
        'pred_prob_max': float(preds.max()),
        'accuracy_at_0.5': round(acc, 4),
        'AUROC': round(auroc, 4) if auroc == auroc else None,
        'AUPRC': round(auprc, 4) if auprc == auprc else None,
        'F1': round(f1, 4),
        'MCC': round(mcc, 4),
        'TN': cm[0][0], 'FP': cm[0][1], 'FN': cm[1][0], 'TP': cm[1][1],
        'runtime_sec': round(elapsed, 1),
    }

    # write per-row CSV (so the user can inspect)
    ckpt_short = ckpt_label.replace('.h5', '').replace('.', '_')
    per_row_fp = os.path.join(
        OUT_DIR, f'amrsweep_{ckpt_short}_{cohort}.csv'
    )
    pd.DataFrame({
        'drug_name': drugs_all, 'species_name': species_all,
        'true_label': labels, 'pred_prob': preds, 'pred_label_at_0.5': binary,
    }).to_csv(per_row_fp, index=False)
    metrics['per_row_csv'] = per_row_fp

    return metrics


# ----------------------- execute sweep -----------------------
master_records = []
sweep_start = time.time()
sweep_log_fp = os.path.join(OUT_DIR, 'amr_sweep_progress.log')

with open(sweep_log_fp, 'w') as logf:
    logf.write(f'AMR sweep starting at {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    logf.flush()

    for ckpt_name, cohorts in SWEEPS:
        ckpt_path = os.path.join(SAVED, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f'\n[SKIP] not found: {ckpt_path}')
            logf.write(f'SKIP missing: {ckpt_name}\n')
            continue
        print(f'\n========== checkpoint: {ckpt_name} ==========')
        logf.write(f'\n========== {ckpt_name} ==========\n')
        try:
            t0 = time.time()
            model.load_weights(ckpt_path)
            print(f'  load_weights ok in {time.time()-t0:.1f}s')
        except Exception as e:
            print(f'  load_weights FAILED: {e}')
            logf.write(f'  load_weights FAILED: {e}\n')
            continue
        for cohort in cohorts:
            print(f'  [{cohort:<14s}] running n={len(cohort_dfs[cohort])} ...')
            try:
                m = run_one_cohort_for_current_weights(cohort, ckpt_name)
                master_records.append(m)
                msg = (f'    acc={m["accuracy_at_0.5"]:.4f}  AUROC={m["AUROC"]}  '
                       f'F1={m["F1"]:.4f}  MCC={m["MCC"]:.4f}  '
                       f'pred_rate={m["pred_rate_at_0.5"]:.3f}  '
                       f'pred_mean={m["pred_prob_mean"]:.3e}  '
                       f'runtime={m["runtime_sec"]}s')
                print(msg)
                logf.write(f'  {cohort}: {msg}\n')
                logf.flush()
            except Exception as e:
                print(f'    ERROR: {e}')
                logf.write(f'  {cohort}: ERROR {e}\n')
                logf.flush()

print(f'\n\nTOTAL sweep time: {(time.time()-sweep_start)/60:.1f} min')

# ----------------------- master metrics CSV -----------------------
if master_records:
    df_master = pd.DataFrame(master_records)
    master_fp = os.path.join(OUT_DIR, 'amr_sweep_master_metrics.csv')
    df_master.to_csv(master_fp, index=False)
    print(f'\nmaster metrics CSV: {master_fp}\n')

    # Pretty print pivot (checkpoint × cohort → accuracy, AUROC)
    print('\n--- ACCURACY MATRIX (checkpoint × cohort) ---')
    pivot_acc = df_master.pivot_table(index='checkpoint', columns='cohort', values='accuracy_at_0.5')
    print(pivot_acc.to_string())
    pivot_acc.to_csv(os.path.join(OUT_DIR, 'amr_sweep_pivot_acc.csv'))

    print('\n--- AUROC MATRIX (checkpoint × cohort) ---')
    pivot_auc = df_master.pivot_table(index='checkpoint', columns='cohort', values='AUROC')
    print(pivot_auc.to_string())
    pivot_auc.to_csv(os.path.join(OUT_DIR, 'amr_sweep_pivot_auroc.csv'))

print('done.')
