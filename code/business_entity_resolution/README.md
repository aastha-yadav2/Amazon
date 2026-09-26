# Amazon ML Challenge 2026: Business Entity Resolution

## 1. Problem Statement
The objective of this challenge is to perform large-scale **Business Entity Resolution** by matching noisy records from Source 2 and Source 3 against a clean, deduplicated reference dataset in Source 1. The primary metric is **Macro $F_{0.5}$**, which places higher weight on precision (minimizing false positive entity merges).

---

## 2. Dataset Structure
Expected input directory layout (`dataset/test/` and `dataset/train/`):
- `test_source1.tsv`: Reference entities (`entity_id`, `business_name`, `business_address`, `country`).
- `test_source2.tsv`: Query entities from Source 2 (`entity_id`, `business_name`, `business_address`, `country`).
- `test_source3.tsv`: Query entities from Source 3 (`entity_id`, `business_name`, `business_address`, `country`).

Output format (`output/` directory):
- `matching_results.tsv`: Tab-separated mapping `source1_entity_id\tmatched_entity_ids` (comma-separated matched IDs from Source 2 and Source 3).
- `candidate_pairs.tsv`: Tab-separated mapping `source1_entity_id\tcandidate_entity_ids` (comma-separated candidate IDs from Source 2 and Source 3).

---

## 3. Preprocessing & Normalization
Record attributes undergo standardized text normalization prior to indexing and feature computation:
- Lowercasing and strip of leading/trailing whitespace.
- Punctuation removal (replacing non-alphanumeric characters with spaces).
- Expansion of common legal entity suffixes (e.g., `ltd` -> `limited`, `pvt` -> `private`, `inc` -> `incorporated`, `co` -> `company`).
- Token whitespace collapsing.

---

## 4. Candidate Generation (Blocking)
To avoid an exhaustive $O(N \times M)$ pair comparison across ~1.73M reference entities and ~10M query records, we employ a multi-key blocking engine (`HybridBlocker`):
- **Exact Normalized Business Name Match**: Hashes normalized business names.
- **Exact Normalized Business Address Match**: Hashes normalized business addresses.

Candidates are indexed using memory-efficient inverted index tables. Candidate pairs are generated whenever a Source 2/Source 3 record shares at least one blocking key with a Source 1 entity.

---

## 5. Feature Engineering
Candidate pairs are evaluated using a 12-dimensional feature vector:
1. `name_jaccard`: Character-level 3-gram Jaccard similarity of normalized business names.
2. `name_token_jaccard`: Word token-level Jaccard similarity of normalized names.
3. `address_jaccard`: Character-level 3-gram Jaccard similarity of normalized addresses.
4. `address_token_jaccard`: Word token-level Jaccard similarity of normalized addresses.
5. `exact_name_match`: Binary flag indicating exact normalized name equality.
6. `exact_address_match`: Binary flag indicating exact normalized address equality.
7. `num_token_overlap`: Count of overlapping numeric tokens (digits/street numbers).
8. `country_match`: Binary flag indicating exact country matching.
9. `name_len_diff`: Absolute length difference between normalized names.
10. `address_len_diff`: Absolute length difference between normalized addresses.
11. `name_prefix_match`: Binary flag indicating whether the first word of the name matches.
12. `address_prefix_match`: Binary flag indicating whether the first token of the address matches.

---

## 6. Machine Learning Model
- **Classifier**: `HistGradientBoostingClassifier` (scikit-learn).
- **Hyperparameters**:
  - `max_iter`: 150
  - `learning_rate`: 0.08
  - `max_leaf_nodes`: 31
  - `min_samples_leaf`: 30
  - `l2_regularization`: 1.0
  - `random_state`: 42
- **Threshold Selection**: Optimal threshold determined via $F_{0.5}$ score optimization. The validated baseline submission uses **threshold = 0.40**.

---

## 7. Prediction & Inference Process
1. Index Source 2 and Source 3 entities into the blocking hash tables.
2. Stream Source 1 entities in memory-bounded chunks (20,000 to 50,000 entities per chunk).
3. Query the blocker for candidate query entity IDs.
4. Extract 12 similarity features per candidate pair.
5. Predict match probability using the trained classifier.
6. Retain candidate pairs with probability $\ge$ threshold.
7. Stream predictions directly to `matching_results.tsv` and `candidate_pairs.tsv`.

---

## 8. How to Run the Solution

### Requirements Installation
```bash
pip install -r requirements.txt
```

### End-to-End Pipeline Execution
To execute model training, feature extraction, test candidate generation, and inference:
```bash
python src/pipeline.py
```

### Training script (optional re-training):
```bash
python src/train_model.py
```

---

## 9. Submission Validation Command
Run the official challenge validator script to verify output format integrity:
```bash
python student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

---

## 10. Important Assumptions & Limitations
- **Precision Preference**: The decision threshold prioritizes high precision to align with Macro $F_{0.5}$ evaluation.
- **Resource Constraints**: High-memory indexing requires streaming batching when processing >10M records.
- **Missing Address Fields**: Entities with empty address fields rely primarily on business name signals.
