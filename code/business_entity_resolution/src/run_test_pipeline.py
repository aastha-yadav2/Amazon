"""
End-to-End Test Pipeline: Generate Enhanced Submission Files.
Streams test datasets in memory-conscious chunks.
Outputs matching_results_enhanced.tsv and candidate_pairs_enhanced.tsv.
Does NOT overwrite any existing submission files.
"""

import sys
import os
import time
import re
import joblib
import subprocess
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# Add src to path
SRC_DIR = r'd:\amazon\code\business_entity_resolution\src'
sys.path.insert(0, SRC_DIR)

from blocking import HybridBlocker
from features import compute_pair_features, FEATURE_COLUMNS


def run_test_pipeline():
    start_time = time.time()
    print("==========================================================")
    print("ENHANCED ENTITY RESOLUTION TEST PIPELINE")
    print("==========================================================")

    # Paths
    test_dir = r'd:\amazon\student_resource\dataset\test'
    output_dir = r'd:\amazon\student_resource\output'
    model_path = os.path.join(SRC_DIR, 'model.joblib')

    s1_path = os.path.join(test_dir, 'test_source1.tsv')
    s2_path = os.path.join(test_dir, 'test_source2.tsv')
    s3_path = os.path.join(test_dir, 'test_source3.tsv')

    # Load trained model
    print(f"Loading model from {model_path}...")
    clf = joblib.load(model_path)
    print("Model loaded successfully.")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Enhanced output filenames (do NOT overwrite existing baseline)
    matching_path = os.path.join(output_dir, 'matching_results_enhanced.tsv')
    candidate_path = os.path.join(output_dir, 'candidate_pairs_enhanced.tsv')

    # Step 1: Index Test Source 2 and Source 3
    print("\n[1/5] Indexing Test Source 2 and Source 3...")
    blocker = HybridBlocker(max_core=250, max_num_name=150, max_name2=150)
    s23_dict = {}

    def index_source_streaming(file_path):
        print(f"  Streaming {file_path}...")
        chunk_count = 0
        for chunk in pd.read_csv(file_path, sep='\t', chunksize=200000):
            blocker.index_dataframe(chunk)
            for eid, name, addr, cty in zip(
                chunk['entity_id'], chunk['business_name'],
                chunk['business_address'], chunk['country']
            ):
                s23_dict[eid] = (name, addr, cty)
            chunk_count += 1
            print(f"    Chunk {chunk_count} indexed ({len(s23_dict):,} total records)...")

    index_source_streaming(s2_path)
    index_source_streaming(s3_path)
    blocker.finalize()

    print(f"  Index complete: {len(s23_dict):,} S2/S3 records indexed.")
    print(f"  Index time: {time.time()-start_time:.1f}s")

    # Step 2: Stream Source 1, generate candidates, extract features, score
    print("\n[2/5] Processing Test Source 1 entities...")
    threshold = 0.55  # Optimal threshold from validation

    total_s1 = 0
    total_candidates = 0
    total_matched_entities = 0
    total_matched_pairs = 0

    with open(matching_path, 'w', encoding='utf-8') as f_match, \
         open(candidate_path, 'w', encoding='utf-8') as f_cand:

        # Write headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        chunk_size = 20000  # Smaller chunks for memory safety
        chunk_num = 0
        for chunk in pd.read_csv(s1_path, sep='\t', chunksize=chunk_size):
            chunk_num += 1
            s1_ids = chunk['entity_id'].values
            names = chunk['business_name'].values
            addrs = chunk['business_address'].values
            countries = chunk['country'].values

            for s1_id, n1, a1, cty1 in zip(s1_ids, names, addrs, countries):
                total_s1 += 1

                # Generate candidates via multi-key blocking
                cands = blocker.get_candidates_for_record(n1, a1)
                total_candidates += len(cands)
                cand_list = list(cands)

                # Write candidate pairs
                f_cand.write(f"{s1_id}\t{','.join(cand_list)}\n")

                if not cand_list:
                    # Singleton - no candidates
                    f_match.write(f"{s1_id}\t\n")
                    continue

                # Extract features and score
                X_cand = []
                for cid in cand_list:
                    c_info = s23_dict.get(cid, ("", "", ""))
                    feats = compute_pair_features(n1, a1, cty1, c_info[0], c_info[1], c_info[2])
                    X_cand.append(feats)

                X_arr = np.array(X_cand, dtype=np.float32)
                probs = clf.predict_proba(X_arr)[:, 1]

                # Filter by threshold
                matched = [cid for cid, p in zip(cand_list, probs) if p >= threshold]

                if matched:
                    total_matched_entities += 1
                    total_matched_pairs += len(matched)
                    f_match.write(f"{s1_id}\t{','.join(matched)}\n")
                else:
                    f_match.write(f"{s1_id}\t\n")

            elapsed = time.time() - start_time
            avg_cands = total_candidates / total_s1 if total_s1 else 0
            print(f"  Chunk {chunk_num}: {total_s1:,} S1 processed | "
                  f"Avg cands: {avg_cands:.1f} | "
                  f"Matched entities: {total_matched_entities:,} | "
                  f"Matched pairs: {total_matched_pairs:,} | "
                  f"Elapsed: {elapsed:.0f}s")

    # Step 3: Summary
    print("\n==========================================================")
    print("TEST PIPELINE SUMMARY")
    print("==========================================================")
    print(f"Total Test S1 Entities      : {total_s1:,}")
    print(f"Total Candidate Pairs       : {total_candidates:,}")
    print(f"Avg Candidates per S1       : {total_candidates/total_s1:.2f}")
    print(f"Threshold Used              : {threshold}")
    print(f"Matched S1 Entities         : {total_matched_entities:,}")
    print(f"Total Matched Pairs         : {total_matched_pairs:,}")
    print(f"Singleton S1 Entities       : {total_s1 - total_matched_entities:,}")
    print(f"Pipeline Runtime            : {time.time()-start_time:.1f}s")
    print(f"Output matching file        : {matching_path}")
    print(f"Output candidate file       : {candidate_path}")
    print("==========================================================")

    # Step 4: Run Official Validator
    print("\n[3/5] Running Official Validator...")
    validator_path = r'd:\amazon\student_resource\utils\validate_submission.py'

    cmd = [
        sys.executable, validator_path,
        '--matching', matching_path,
        '--candidate', candidate_path,
        '--test-dir', test_dir
    ]
    print(f"  Command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
    print(f"  Validator exit code: {res.returncode}")

    # Step 5: Copy to standard output names for final packaging (only if validation passed)
    if res.returncode == 0:
        print("\n[4/5] Validation PASSED. Copying to standard output names...")
        import shutil
        std_matching = os.path.join(output_dir, 'matching_results.tsv')
        std_candidate = os.path.join(output_dir, 'candidate_pairs.tsv')
        shutil.copy2(matching_path, std_matching)
        shutil.copy2(candidate_path, std_candidate)
        print(f"  Copied to: {std_matching}")
        print(f"  Copied to: {std_candidate}")
        print("\n[5/5] Enhanced submission files are ready for leaderboard upload.")
    else:
        print("\n[4/5] Validation FAILED. NOT copying to standard output names.")
        print("  Fix validation errors before proceeding.")

    print(f"\nTotal pipeline time: {time.time()-start_time:.1f}s")


if __name__ == '__main__':
    run_test_pipeline()
