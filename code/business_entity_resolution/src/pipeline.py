"""
End-to-End Pipeline Execution Script.
Processes full test dataset in low-RAM streaming chunks:
1. Builds candidate index on test_source2.tsv and test_source3.tsv.
2. Generates candidate pairs for all test_source1.tsv entities.
3. Extracts 12-dimensional features and scores candidates with trained model.
4. Outputs matching_results.tsv and candidate_pairs.tsv in tab-separated format.
5. Executes official submission validator script.
"""

import sys
import os
import time
import re
import joblib
import subprocess
import pandas as pd
import numpy as np

# Import custom modules
from blocking import HybridBlocker
from features import compute_pair_features


def run_pipeline(test_dir="dataset/test", output_dir="output", threshold=0.45):
    start_time = time.time()
    print("==========================================================")
    print("STARTING END-TO-END ENTITY RESOLUTION TEST PIPELINE")
    print("==========================================================")

    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train_model.py first!")

    clf = joblib.load(model_path)
    print("Loaded trained model successfully.")

    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    # Step 1: Index Test Source 2 and Source 3
    print("\n[1/4] Indexing Test Source 2 and Source 3 in Blocker...")
    blocker = HybridBlocker(max_core=250, max_num_name=150, max_name2=150)
    
    # Store attribute dictionary for S2/S3 (name, address, country)
    s23_dict = {}

    def index_source_file(file_path):
        print(f"  Reading {file_path}...")
        for chunk in pd.read_csv(file_path, sep='\t', chunksize=200000, usecols=['entity_id', 'business_name', 'business_address', 'country']):
            blocker.index_dataframe(chunk)
            for eid, name, addr, cty in zip(chunk['entity_id'], chunk['business_name'], chunk['business_address'], chunk['country']):
                s23_dict[eid] = (name, addr, cty)

    index_source_file(s2_path)
    index_source_file(s3_path)
    blocker.finalize()
    print(f"  Test source index ready ({len(s23_dict):,} records indexed).")

    # Step 2: Stream Test Source 1, generate candidates, extract features, predict matches
    print("\n[2/4] Generating Candidates & Scoring Test Source 1...")
    
    total_s1 = 0
    total_candidates = 0
    total_matched_entities = 0

    with open(matching_path, 'w', encoding='utf-8') as f_match, \
         open(candidate_path, 'w', encoding='utf-8') as f_cand:
        
        # Write headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        chunk_size = 50000
        for chunk in pd.read_csv(s1_path, sep='\t', chunksize=chunk_size):
            s1_ids = chunk['entity_id'].values
            names = chunk['business_name'].values
            addrs = chunk['business_address'].values
            countries = chunk['country'].values

            for s1_id, n1, a1, cty1 in zip(s1_ids, names, addrs, countries):
                total_s1 += 1
                cands = blocker.get_candidates_for_record(n1, a1)
                total_candidates += len(cands)

                cand_list = list(cands)
                f_cand.write(f"{s1_id}\t{','.join(cand_list)}\n")

                if not cand_list:
                    # Singleton - no match
                    f_match.write(f"{s1_id}\t\n")
                    continue

                # Batch feature extraction for candidates
                X_cand = []
                for cid in cand_list:
                    c_info = s23_dict.get(cid, ("", "", ""))
                    feats = compute_pair_features(n1, a1, cty1, c_info[0], c_info[1], c_info[2])
                    X_cand.append(feats)

                X_arr = np.array(X_cand, dtype=np.float32)
                probs = clf.predict_proba(X_arr)[:, 1]

                # Filter predicted matches based on threshold
                matched_cands = [cid for cid, p in zip(cand_list, probs) if p >= threshold]

                if matched_cands:
                    total_matched_entities += 1
                    f_match.write(f"{s1_id}\t{','.join(matched_cands)}\n")
                else:
                    f_match.write(f"{s1_id}\t\n")

            print(f"  Processed {total_s1:,} S1 entities (avg candidates: {total_candidates/total_s1:.2f})...")

    print("\n----------------------------------------------------------")
    print("PIPELINE TEST RUN SUMMARY")
    print("----------------------------------------------------------")
    print(f"Total Test S1 Entities     : {total_s1:,}")
    print(f"Total Test Candidate Pairs  : {total_candidates:,}")
    print(f"Avg Candidates per S1       : {total_candidates/total_s1:.2f}")
    print(f"Total Matched S1 Entities   : {total_matched_entities:,}")
    print("----------------------------------------------------------")

    # Step 3: Run Validator
    print("\n[3/4] Running Official Validator...")
    validator_script = os.path.join(os.path.dirname(__file__), "..", "..", "utils", "validate_submission.py")
    if os.path.exists(validator_script):
        cmd = [
            sys.executable, validator_script,
            "--matching", matching_path,
            "--candidate", candidate_path,
            "--test-dir", test_dir
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print("Validator Errors:")
            print(res.stderr)
    else:
        print(f"Validator script not found at {validator_script}.")

    print(f"\nPipeline execution finished in {time.time()-start_time:.1f}s.")


if __name__ == '__main__':
    run_pipeline()
