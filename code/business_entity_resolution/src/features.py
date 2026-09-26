"""
12-Dimensional Vector Feature Extraction Engine for Entity Resolution.
Computes rich similarity signals between Source 1 records and candidate S2/S3 records.
"""

import re
import difflib
import numpy as np
import pandas as pd

STOPWORDS = {
    'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'co', 'company',
    'pvt', 'private', 'services', 'service', 'group', 'enterprises', 'enterprise',
    'solutions', 'the', 'and', 'of', 'in', 'at', 'store', 'shop', 'india', 'us', 'usa',
    'center', 'centre', 'industries', 'international', 'systems', 'tech', 'technologies'
}

def get_tokens(s):
    if not isinstance(s, str) or pd.isna(s):
        return []
    return re.sub(r'[^a-z0-9]', ' ', str(s).lower()).split()

def get_core_tokens(s):
    return [t for t in get_tokens(s) if t not in STOPWORDS]

def get_numbers(s):
    if not isinstance(s, str) or pd.isna(s):
        return set()
    return set(re.findall(r'\b\d+\b', str(s)))

def jaccard(set1, set2):
    if not set1 and not set2:
        return 0.0
    union = len(set1 | set2)
    return len(set1 & set2) / union if union > 0 else 0.0

def containment(set1, set2):
    if not set1 or not set2:
        return 0.0
    min_len = min(len(set1), len(set2))
    return len(set1 & set2) / min_len if min_len > 0 else 0.0

def char_fuzzy_ratio(s1, s2):
    str1 = str(s1).lower() if pd.notna(s1) else ""
    str2 = str(s2).lower() if pd.notna(s2) else ""
    if not str1 and not str2:
        return 0.0
    # Use character 3-gram Jaccard for ultra-fast fuzzy similarity
    if len(str1) < 3 or len(str2) < 3:
        return 1.0 if str1 == str2 else 0.0
    g1 = set(str1[i:i+3] for i in range(len(str1)-2))
    g2 = set(str2[i:i+3] for i in range(len(str2)-2))
    return jaccard(g1, g2)


def compute_pair_features(n1, a1, cty1, n2, a2, cty2):
    """
    Compute 12 matching features for a pair of entities (S1 vs S2/S3).
    """
    toks1 = set(get_tokens(n1))
    toks2 = set(get_tokens(n2))
    
    core1 = set(get_core_tokens(n1))
    core2 = set(get_core_tokens(n2))
    
    addr_toks1 = set(get_tokens(a1))
    addr_toks2 = set(get_tokens(a2))
    
    nums1 = get_numbers(a1)
    nums2 = get_numbers(a2)
    
    # 1. name_jaccard_token
    f_name_jaccard = jaccard(toks1, toks2)
    
    # 2. name_jaccard_core
    f_name_core_jaccard = jaccard(core1, core2)
    
    # 3. name_token_containment
    f_containment = containment(toks1, toks2)
    
    # 4. name_levenshtein_ratio (char 3-gram fuzzy similarity)
    f_fuzzy_ratio = char_fuzzy_ratio(n1, n2)
    
    # 5. address_jaccard_token
    f_addr_jaccard = jaccard(addr_toks1, addr_toks2)
    
    # 6. address_number_overlap
    num_overlap_count = len(nums1 & nums2)
    
    # 7. address_number_mismatch
    num_mismatch_count = len(nums1 ^ nums2) if (nums1 and nums2) else 0
    
    # 8. exact_name
    exact_n = 1.0 if (n1 and n2 and "".join(toks1) == "".join(toks2)) else 0.0
    
    # 9. exact_address
    exact_a = 1.0 if (a1 and a2 and "".join(addr_toks1) == "".join(addr_toks2)) else 0.0
    
    # 10. country_match
    cty1_s = str(cty1).strip().lower() if pd.notna(cty1) else ""
    cty2_s = str(cty2).strip().lower() if pd.notna(cty2) else ""
    cty_match = 1.0 if (not cty1_s or not cty2_s or cty1_s == cty2_s) else 0.0
    
    # 11. name_prefix_match
    p1 = get_tokens(n1)[:1]
    p2 = get_tokens(n2)[:1]
    prefix_m = 1.0 if (p1 and p2 and p1[0] == p2[0]) else 0.0
    
    # 12. overall_composite_sim
    composite = 0.4 * f_name_core_jaccard + 0.3 * f_addr_jaccard + 0.3 * f_fuzzy_ratio
    
    return [
        f_name_jaccard, f_name_core_jaccard, f_containment, f_fuzzy_ratio,
        f_addr_jaccard, num_overlap_count, num_mismatch_count,
        exact_n, exact_a, cty_match, prefix_m, composite
    ]

FEATURE_COLUMNS = [
    'name_jaccard_token', 'name_jaccard_core', 'name_token_containment', 'name_fuzzy_ratio',
    'address_jaccard_token', 'address_number_overlap', 'address_number_mismatch',
    'exact_name', 'exact_address', 'country_match', 'name_prefix_match', 'overall_composite_sim'
]
