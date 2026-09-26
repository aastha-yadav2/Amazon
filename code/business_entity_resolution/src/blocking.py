"""
Multi-Key High-Recall Hybrid Candidate Blocking Engine for Entity Resolution.
Combines exact matching with token-based and structural address blocking key indexing.
Includes frequency-capping to prevent candidate explosion on common business terms.
"""

import re
import pandas as pd
import numpy as np

STOPWORDS = {
    'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'co', 'company',
    'pvt', 'private', 'services', 'service', 'group', 'enterprises', 'enterprise',
    'solutions', 'the', 'and', 'of', 'in', 'at', 'store', 'shop', 'india', 'us', 'usa',
    'center', 'centre', 'industries', 'international', 'systems', 'tech', 'technologies'
}

def norm_exact(s):
    if not isinstance(s, str) or pd.isna(s):
        return ""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def norm_tokens(s):
    if not isinstance(s, str) or pd.isna(s):
        return []
    return [t for t in re.sub(r'[^a-z0-9]', ' ', str(s).lower()).split() if t not in STOPWORDS]

def extract_blocking_keys(name, addr):
    """
    Extract multi-key representations for candidate indexing:
    1. exact_name: alphanumeric-only normalized name
    2. exact_addr: alphanumeric-only normalized address
    3. core_name: sorted non-stopword core tokens of business name
    4. num_name: street/house number + first core name token
    5. name_2tok: first 2 non-stopword tokens of business name
    """
    n_tokens = norm_tokens(name)
    addr_str = str(addr) if pd.notna(addr) else ""
    
    # 1. Exact Name
    exact_name = "".join(n_tokens)
    
    # 2. Exact Address
    a_tokens = norm_tokens(addr)
    exact_addr = "".join(a_tokens)
    
    # 3. Core Name (Sorted)
    core_tokens = sorted([t for t in n_tokens if len(t) > 1])
    core_name = " ".join(core_tokens) if len(core_tokens) >= 1 else ""
    
    # 4. Street/House Number + First Core Name Token
    num_match = re.search(r'\b\d+\b', addr_str)
    street_num = num_match.group(0) if num_match else ""
    first_core = [t for t in n_tokens if len(t) > 2][:1]
    num_name = f"{street_num}#{first_core[0]}" if (street_num and first_core) else ""
    
    # 5. First 2 Core Tokens
    name_2tok = " ".join(n_tokens[:2]) if len(n_tokens) >= 2 else ""
    
    return exact_name, exact_addr, core_name, num_name, name_2tok


class HybridBlocker:
    def __init__(self, max_core=250, max_num_name=150, max_name2=150):
        self.max_core = max_core
        self.max_num_name = max_num_name
        self.max_name2 = max_name2
        
        self.exact_name_index = {}
        self.exact_addr_index = {}
        self.core_name_index = {}
        self.num_name_index = {}
        self.name2_index = {}

    def index_dataframe(self, df):
        """Index a DataFrame containing entity_id, business_name, business_address."""
        eids = df['entity_id'].values
        names = df['business_name'].values
        addrs = df['business_address'].values
        
        for eid, name, addr in zip(eids, names, addrs):
            k_n, k_a, core_n, k_nn, k_n2 = extract_blocking_keys(name, addr)
            
            if k_n:
                self.exact_name_index.setdefault(k_n, []).append(eid)
            if k_a:
                self.exact_addr_index.setdefault(k_a, []).append(eid)
            if core_n and len(core_n) > 3:
                self.core_name_index.setdefault(core_n, []).append(eid)
            if k_nn:
                self.num_name_index.setdefault(k_nn, []).append(eid)
            if k_n2 and len(k_n2) > 4:
                self.name2_index.setdefault(k_n2, []).append(eid)

    def finalize(self):
        """Apply frequency capping filter to eliminate ultra-frequent noisy blocks."""
        self.filt_core = {k: v for k, v in self.core_name_index.items() if len(v) <= self.max_core}
        self.filt_num_name = {k: v for k, v in self.num_name_index.items() if len(v) <= self.max_num_name}
        self.filt_name2 = {k: v for k, v in self.name2_index.items() if len(v) <= self.max_name2}

    def get_candidates_for_record(self, name, addr):
        """Query inverted indexes for a single S1 record."""
        k_n, k_a, core_n, k_nn, k_n2 = extract_blocking_keys(name, addr)
        cands = set()
        
        if k_n in self.exact_name_index:
            cands.update(self.exact_name_index[k_n])
        if k_a in self.exact_addr_index:
            cands.update(self.exact_addr_index[k_a])
        if core_n in self.filt_core:
            cands.update(self.filt_core[core_n])
        if k_nn in self.filt_num_name:
            cands.update(self.filt_num_name[k_nn])
        if k_n2 in self.filt_name2:
            cands.update(self.filt_name2[k_n2])
            
        return cands
