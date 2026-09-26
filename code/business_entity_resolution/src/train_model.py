"""
Model Training, Validation, and Macro F0.5 Threshold Optimization Module.
Includes ground-truth candidate recall benchmark, train/val split at S1 level,
HistGradientBoostingClassifier training, precision/recall metrics, and error analysis.
"""

import sys
import os
import time
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding='utf-8')

# Import custom modules
from blocking import HybridBlocker
from features import compute_pair_features, FEATURE_COLUMNS


def calculate_macro_f05(ground_truth_dict, predicted_dict, all_s1_ids):
    """
    Calculate Macro F0.5 score across all S1 entities (including singletons).
    """
    f05_scores = []
    precisions = []
    recalls = []

    for s1_id in all_s1_ids:
        true_set = ground_truth_dict.get(s1_id, set())
        pred_set = predicted_dict.get(s1_id, set())

        if not true_set and not pred_set:
            # Correct singleton prediction
            f05_scores.append(1.0)
            precisions.append(1.0)
            recalls.append(1.0)
        elif not true_set and pred_set:
            # False positive on singleton
            f05_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(1.0)
        elif true_set and not pred_set:
            # Missed matches on non-singleton
            f05_scores.append(0.0)
            precisions.append(1.0)
            recalls.append(0.0)
        else:
            # Both true_set and pred_set exist
            hits = len(true_set & pred_set)
            prec = hits / len(pred_set) if len(pred_set) > 0 else 0.0
            rec = hits / len(true_set) if len(true_set) > 0 else 0.0

            precisions.append(prec)
            recalls.append(rec)

            if prec + rec > 0:
                f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
            else:
                f05 = 0.0
            f05_scores.append(f05)

    return np.mean(f05_scores), np.mean(precisions), np.mean(recalls)


def train_and_evaluate():
    start_time = time.time()
    print("==========================================================")
    print("STEP 1: Loading Training Data & Ground Truth Benchmark")
    print("==========================================================")

    data_dir = r'd:\amazon\student_resource\dataset\train'
    gt_path = os.path.join(data_dir, 'train_ground_truth.tsv')
    s1_path = os.path.join(data_dir, 'train_source1.tsv')
    s2_path = os.path.join(data_dir, 'train_source2.tsv')
    s3_path = os.path.join(data_dir, 'train_source3.tsv')

    gt = pd.read_csv(gt_path, sep='\t')
    print(f"Loaded train ground truth: {len(gt):,} rows.")

    # Sample 100,000 S1 entities for training & validation
    sample_gt = gt.sample(n=100000, random_state=42).copy()
    sample_s1_ids = set(sample_gt['source1_entity_id'])

    gt_dict = {}
    total_true_pairs = 0
    for s1, mids in zip(sample_gt['source1_entity_id'], sample_gt['matched_entity_ids']):
        if pd.isna(mids) or not str(mids).strip():
            gt_dict[s1] = set()
        else:
            s = set(str(mids).split(','))
            gt_dict[s1] = s
            total_true_pairs += len(s)

    print(f"Benchmark sample: 100,000 S1 entities containing {total_true_pairs:,} true ground-truth pairs.")

    # Load Source Records
    s1_df = pd.read_csv(s1_path, sep='\t')
    s1_sub = s1_df[s1_df['entity_id'].isin(sample_s1_ids)].copy()

    s2_df = pd.read_csv(s2_path, sep='\t')
    s3_df = pd.read_csv(s3_path, sep='\t')

    # Index Source 2 and Source 3 in Blocker
    print("\nIndexing Source 2 & Source 3 in Hybrid Candidate Blocker...")
    blocker = HybridBlocker(max_core=250, max_num_name=150, max_name2=150)
    blocker.index_dataframe(s2_df)
    blocker.index_dataframe(s3_df)
    blocker.finalize()
    print("Blocker index built successfully.")

    # Generate Candidate Pairs & Measure Recall
    print("\nGenerating candidate pairs for S1 sample...")
    cand_pairs = []
    hits = 0
    s23_dict = {}

    # Build lookup dict for S2 and S3 attributes
    for df in [s2_df, s3_df]:
        for eid, name, addr, cty in zip(df['entity_id'], df['business_name'], df['business_address'], df['country']):
            s23_dict[eid] = (name, addr, cty)

    total_candidates_generated = 0

    for s1_id, n1, a1, cty1 in zip(s1_sub['entity_id'], s1_sub['business_name'], s1_sub['business_address'], s1_sub['country']):
        cands = blocker.get_candidates_for_record(n1, a1)
        total_candidates_generated += len(cands)
        true_set = gt_dict[s1_id]

        if true_set:
            hits += len(cands & true_set)

        for cid in cands:
            cand_pairs.append((s1_id, cid, n1, a1, cty1))

    cand_recall = hits / total_true_pairs if total_true_pairs > 0 else 0.0
    avg_cands_per_s1 = total_candidates_generated / len(s1_sub)

    print("\n----------------------------------------------------------")
    print("GROUND TRUTH CANDIDATE RECALL REPORT")
    print("----------------------------------------------------------")
    print(f"Total S1 Benchmark Entities : {len(s1_sub):,}")
    print(f"Total True Pairs            : {total_true_pairs:,}")
    print(f"Total Candidates Generated  : {total_candidates_generated:,}")
    print(f"Average Candidates per S1   : {avg_cands_per_s1:.2f}")
    print(f"Candidate Recall on GT      : {cand_recall:.4f} ({cand_recall*100:.2f}%)")
    print("----------------------------------------------------------")

    # Extract Features & Labels
    print("\nExtracting 12-dimensional features for candidate pairs...")
    X_rows = []
    y_labels = []
    s1_id_list = []
    cand_id_list = []

    for s1_id, cid, n1, a1, cty1 in cand_pairs:
        c_info = s23_dict.get(cid)
        if not c_info:
            continue
        n2, a2, cty2 = c_info
        
        feats = compute_pair_features(n1, a1, cty1, n2, a2, cty2)
        label = 1 if cid in gt_dict.get(s1_id, set()) else 0

        X_rows.append(feats)
        y_labels.append(label)
        s1_id_list.append(s1_id)
        cand_id_list.append(cid)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)
    s1_ids_arr = np.array(s1_id_list)
    cand_ids_arr = np.array(cand_id_list)

    print(f"Extracted feature matrix X shape: {X.shape}, Positives: {np.sum(y):,} ({np.mean(y)*100:.2f}%)")

    # Train / Validation Split at S1 entity level
    print("\n==========================================================")
    print("STEP 2: Entity-Level Train / Validation Split & Training")
    print("==========================================================")
    unique_s1s = np.array(list(set(s1_ids_arr)))
    train_s1s, val_s1s = train_test_split(unique_s1s, test_size=0.20, random_state=42)

    train_mask = np.isin(s1_ids_arr, train_s1s)
    val_mask = np.isin(s1_ids_arr, val_s1s)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"Train samples: {len(X_train):,}, Validation samples: {len(X_val):,}")

    # Train HistGradientBoostingClassifier
    print("\nTraining HistGradientBoostingClassifier...")
    clf = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.08,
        max_depth=8,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42
    )
    clf.fit(X_train, y_train)

    val_probs = clf.predict_proba(X_val)[:, 1]

    # Evaluate & Optimize Threshold for S1-Level Macro F0.5
    print("\n==========================================================")
    print("STEP 3: Threshold Tuning for S1-Level Macro F0.5")
    print("==========================================================")
    val_s1_ids_set = set(val_s1s)
    val_s1_arr = s1_ids_arr[val_mask]
    val_cand_arr = cand_ids_arr[val_mask]

    best_thresh = 0.50
    best_f05 = 0.0
    best_prec = 0.0
    best_rec = 0.0
    best_pred_dict = {}

    thresholds_to_test = np.linspace(0.20, 0.85, 14)

    for th in thresholds_to_test:
        pred_dict = {}
        for s1, cid, prob in zip(val_s1_arr, val_cand_arr, val_probs):
            if prob >= th:
                pred_dict.setdefault(s1, set()).add(cid)

        f05, prec, rec = calculate_macro_f05(gt_dict, pred_dict, val_s1_ids_set)
        print(f"Threshold: {th:.2f} -> Macro F0.5: {f05:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = th
            best_prec = prec
            best_rec = rec
            best_pred_dict = pred_dict

    print("\n----------------------------------------------------------")
    print("OPTIMAL VALIDATION RESULTS")
    print("----------------------------------------------------------")
    print(f"Selected Threshold          : {best_thresh:.2f}")
    print(f"Validation Macro F0.5       : {best_f05:.4f}")
    print(f"Validation Macro Precision  : {best_prec:.4f}")
    print(f"Validation Macro Recall     : {best_rec:.4f}")
    
    num_matched_s1 = sum(1 for s in best_pred_dict.values() if len(s) > 0)
    print(f"Matched S1 Entities in Val  : {num_matched_s1:,} / {len(val_s1_ids_set):,}")
    print("----------------------------------------------------------")

    # Save Model Artifact
    os.makedirs(r'd:\amazon\code\business_entity_resolution\src', exist_ok=True)
    joblib.dump(clf, r'd:\amazon\code\business_entity_resolution\src\model.joblib')
    print("Saved trained model to src/model.joblib.")

    # False Positive / False Negative Error Analysis
    print("\n==========================================================")
    print("STEP 4: False Positive / False Negative Error Analysis")
    print("==========================================================")
    fp_cases = []
    fn_cases = []

    for s1 in val_s1_ids_set:
        true_s = gt_dict.get(s1, set())
        pred_s = best_pred_dict.get(s1, set())

        # False Positives
        fps = pred_s - true_s
        if fps and len(fp_cases) < 5:
            fp_cases.append((s1, list(fps)[:2]))

        # False Negatives
        fns = true_s - pred_s
        if fns and len(fn_cases) < 5:
            fn_cases.append((s1, list(fns)[:2]))

    print("\nSample False Positive Errors (Model predicted match, Ground Truth is negative):")
    for s1, fps in fp_cases:
        s1_row = s1_df[s1_df['entity_id'] == s1].iloc[0]
        print(f"  S1 ({s1}) Name: '{s1_row['business_name']}' | Addr: '{s1_row['business_address']}'")
        for fp in fps:
            c_info = s23_dict.get(fp, ("Unknown", "Unknown", ""))
            print(f"    -> FP ({fp}) Name: '{c_info[0]}' | Addr: '{c_info[1]}'")

    print("\nSample False Negative Errors (Ground Truth had match, Model missed or candidate missing):")
    for s1, fns in fn_cases:
        s1_row = s1_df[s1_df['entity_id'] == s1].iloc[0]
        print(f"  S1 ({s1}) Name: '{s1_row['business_name']}' | Addr: '{s1_row['business_address']}'")
        for fn in fns:
            c_info = s23_dict.get(fn, ("Unknown", "Unknown", ""))
            print(f"    -> FN ({fn}) Name: '{c_info[0]}' | Addr: '{c_info[1]}'")

    print(f"\nTraining & evaluation completed in {time.time()-start_time:.1f}s.")
    return best_thresh


if __name__ == '__main__':
    train_and_evaluate()
