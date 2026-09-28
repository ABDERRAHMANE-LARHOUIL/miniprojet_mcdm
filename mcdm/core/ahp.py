import numpy as np
from typing import Dict, Tuple

SAATY_RI: Dict[int, float] = {
    1: 0.0,
    2: 0.0,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.56
}

def compute_ahp_weights(pairwise_matrix: np.ndarray) -> Tuple[np.ndarray, float, float, bool]:
    """
    Calcule les poids AHP par la méthode approximative de normalisation par colonne
    et évalue la cohérence (CR < 0.1).
    """
    n = pairwise_matrix.shape[0]
    
    col_sums = pairwise_matrix.sum(axis=0)
    normalized_mat = pairwise_matrix / col_sums
    weights = normalized_mat.mean(axis=1)
    
    lambda_max = float(np.dot(col_sums, weights))
    
    if n <= 2:
        return weights, 0.0, 0.0, True
        
    ci = (lambda_max - n) / (n - 1)
    ri = SAATY_RI.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    
    is_consistent = cr < 0.10
    return weights, ci, cr, is_consistent