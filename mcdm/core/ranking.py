import numpy as np
import pandas as pd
from typing import List

def run_wsm(
    decision_matrix: np.ndarray,
    weights: np.ndarray,
    criteria_types: List[str]
) -> pd.DataFrame:
    """Méthode des sommes pondérées (Weighted Sum Method)."""
    m, n = decision_matrix.shape
    norm_mat = np.zeros_like(decision_matrix, dtype=float)
    
    for j in range(n):
        col = decision_matrix[:, j]
        if criteria_types[j] == "Bénéfice (+)":
            col_max = np.max(col)
            norm_mat[:, j] = col / col_max if col_max != 0 else 0
        else:
            col_min = np.min(col)
            norm_mat[:, j] = col_min / col if not np.any(col == 0) else 0
            
    scores = np.dot(norm_mat, weights)
    return scores

def run_topsis(
    decision_matrix: np.ndarray,
    weights: np.ndarray,
    criteria_types: List[str]
) -> pd.DataFrame:
    """Méthode TOPSIS classique."""
    m, n = decision_matrix.shape
    
    # 1. Normalisation vectorielle
    sq_sums = np.sqrt(np.sum(decision_matrix ** 2, axis=0))
    norm_mat = decision_matrix / sq_sums
    
    # 2. Matrice pondérée
    weighted_mat = norm_mat * weights
    
    # 3. Idéal positif et négatif
    v_pos = np.zeros(n)
    v_neg = np.zeros(n)
    
    for j in range(n):
        if criteria_types[j] == "Bénéfice (+)":
            v_pos[j] = np.max(weighted_mat[:, j])
            v_neg[j] = np.min(weighted_mat[:, j])
        else:
            v_pos[j] = np.min(weighted_mat[:, j])
            v_neg[j] = np.max(weighted_mat[:, j])
            
    # 4. Mesures de distance
    d_pos = np.sqrt(np.sum((weighted_mat - v_pos) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weighted_mat - v_neg) ** 2, axis=1))
    
    # 5. Proximité relative RC
    denom = d_pos + d_neg
    rc = np.where(denom == 0, 0, d_neg / denom)
    return rc