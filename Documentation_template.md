# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Silver Strikers

**Team Members:** Aastha Yadav & Team

**Submission Date:** 26 September 2026

---

## 1. Executive Summary

We developed a scalable hybrid business entity resolution pipeline for matching noisy records from Source 2 and Source 3 against the deduplicated reference entities in Source 1. The solution combines normalization-based blocking with machine-learning-based pair scoring using name, address, numerical-token, and exact-name similarity features, followed by threshold-based matching optimized for precision.

The validated baseline submission achieved an official **Public Leaderboard Macro $F_{0.5}$ Score of 0.478699** (displayed as **0.479**).

---

## 2. Methodology

### 2.1 Problem Analysis

The dataset contains a clean deduplicated reference source (Source 1) and two noisy sources (Source 2 and Source 3). The major challenges observed were variations in business names and addresses, differences in formatting, token ordering, punctuation, and noisy records.

The solution focuses on normalized textual representations and similarity-based comparison rather than relying strictly on exact string equality.

The evaluation metric is Macro $F_{0.5}$, which places greater importance on precision ($P$) over recall ($R$). Therefore, incorrect entity merges (false positives) are heavily penalized during threshold selection.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Machine Learning Classifier

**Core Innovation:** A scalable candidate-generation and pair-scoring pipeline that first reduces the search space using exact normalized business-name or address signals and then applies a machine-learning classifier to estimate the likelihood that a candidate pair represents the same business entity.

The pipeline consists of:
1. Data normalization and preprocessing.
2. Candidate generation using normalized business name and address.
3. Feature extraction for candidate pairs.
4. Machine-learning-based pair scoring.
5. Threshold-based selection of matched entities.
6. Generation and validation of the final matching output.

---

## 3. Candidate Generation (Blocking)

Candidate generation was used to avoid performing an all-pairs comparison between the large source datasets and the Source 1 reference dataset.

### Blocking keys used:
- Normalized business name
- Normalized business address
- Exact normalized name match
- Exact normalized address match

Records were normalized before blocking to reduce the impact of formatting differences.

The final candidate-generation stage produced approximately:
- **11,598,976 total candidate pairs**
- **5,383,037** candidate pairs from Source 2
- **6,215,939** candidate pairs from Source 3

Candidate generation reduced the comparison space substantially compared with an exhaustive comparison of all records (~1.34M S1 entities covered out of 1.73M).

### Ensuring True Matches are Retained:
The blocking strategy used multiple signals rather than relying on a single field. A candidate was generated when either the normalized business name or normalized business address provided an exact blocking signal.

This allowed the matching model to subsequently evaluate multiple similarity features instead of making the final matching decision directly from the blocking key.

The generated candidate list was retained separately as `candidate_pairs.tsv` for validation and reproducibility.

---

## 4. Matching Model

### Features used:
- **Name features:** Name Jaccard similarity (3-gram character & word token), exact normalized name match.
- **Address features:** Address Jaccard similarity (3-gram character & word token), exact normalized address match.
- **Other features:** Numerical-token overlap between records, country matching, length differences, prefix matching.

### Model type:
**HistGradientBoostingClassifier**

The classifier was trained on labelled candidate pairs generated from the training data.

The model configuration used:
- `max_iter = 150`
- `learning_rate = 0.08`
- `max_leaf_nodes = 31`
- `min_samples_leaf = 30`
- `l2_regularization = 1.0`
- `random_state = 42`

### Threshold selection method:
A validation-based threshold sweep was performed using $F_{0.5}$ as the evaluation metric.

Because $F_{0.5}$ emphasizes precision, the threshold was selected with the objective of reducing incorrect entity merges while retaining genuine matches.

For the validated final public leaderboard submission, multiple thresholds were evaluated:
- Threshold 0.70: Macro $F_{0.5} = 0.469$
- **Threshold 0.40: Public Leaderboard Score = 0.478699 (Selected Final Submission)**

---

## 5. Results & Error Analysis

### Official Leaderboard Result:
* **Public Leaderboard Macro $F_{0.5}$ Score:** **0.478699**
* **Displayed Leaderboard Score:** **0.479**
* **Team:** Silver Strikers
* **Submission Date:** 26 September 2026
* **Selected Threshold:** **0.40**

> [!NOTE]
> The score **0.478699** is the official **PUBLIC LEADERBOARD SCORE** achieved by the submitted baseline output files (`matching_results.tsv` / `candidate_pairs.tsv`).

### Common false positives (wrong merges):
Potential false positives arise when different businesses have highly similar or identical names and overlapping address tokens. This is particularly challenging when common business names occur across multiple entities (e.g., generic store names).

### Common false negatives (missed matches):
Potential false negatives arise when the same business is represented with substantial changes in both its business name and address, leaving insufficient overlap with the blocking keys.

---

## 6. Experimental Enhancements (Offline / Unsubmitted)

In addition to the validated submission, an enhanced hybrid candidate blocking engine was prototyped offline featuring:
- Multi-key high-recall candidate blocking (core tokens, numeric tokens, multi-word n-grams).
- 12-dimensional feature set including Jaccard token/character metrics.
- S1-level validation split achieving **75.13% offline candidate recall** and **0.8203 offline validation $F_{0.5}$** at threshold 0.55.

> [!IMPORTANT]
> The enhanced pipeline figures above represent **offline validation metrics on training data only**. They have **not** been submitted or verified on the public leaderboard. The official score for our solution remains **0.478699**.

---

## 7. Conclusion

The developed solution combines scalable blocking with machine-learning-based entity matching to handle millions of noisy business records efficiently. The final pipeline generated approximately 11.6 million candidate pairs and used name, address, numerical-token, and exact-name signals for pair classification.

The approach achieved a public leaderboard $F_{0.5}$ score of **0.478699**, demonstrating the effectiveness of combining candidate reduction with supervised pair scoring for large-scale business entity resolution.

---

## Appendix: Code Artefacts

The complete runnable code is included in the submission zip under:

`code/business_entity_resolution/`

The expected structure is:

```text
code/
└── business_entity_resolution/
    ├── src/
    │   ├── blocking.py
    │   ├── features.py
    │   ├── train_model.py
    │   ├── pipeline.py
    │   ├── run_test_pipeline.py
    │   └── package_submission.py
    ├── README.md
    └── requirements.txt
```
