import numpy as np
from scipy.optimize import linprog
from typing import Tuple

def compute_bwm_weights(
    num_criteria: int,
    best_idx: int,
    worst_idx: int,
    best_to_others: np.ndarray,
    others_to_worst: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Résout le problème d'optimisation linéaire BWM :
    min xi
    s.c.
      |w_B - a_Bj * w_j| <= xi
      |w_j - a_jW * w_W| <= xi
      sum(w_j) = 1, w_j >= 0, xi >= 0
    """
    n = num_criteria
    num_vars = n + 1  # [w_1, ..., w_n, xi]
    
    c = np.zeros(num_vars)
    c[-1] = 1.0
    
    A_ub = []
    b_ub = []
    
    # 1. |w_B - a_Bj * w_j| <= xi
    for j in range(n):
        if j != best_idx:
            a_bj = float(best_to_others[j])
            row1 = np.zeros(num_vars)
            row1[best_idx] = 1.0
            row1[j] = -a_bj
            row1[-1] = -1.0
            A_ub.append(row1)
            b_ub.append(0.0)
            
            row2 = np.zeros(num_vars)
            row2[best_idx] = -1.0
            row2[j] = a_bj
            row2[-1] = -1.0
            A_ub.append(row2)
            b_ub.append(0.0)
            
    # 2. |w_j - a_jW * w_W| <= xi
    for j in range(n):
        if j != worst_idx:
            a_jw = float(others_to_worst[j])
            row1 = np.zeros(num_vars)
            row1[j] = 1.0
            row1[worst_idx] = -a_jw
            row1[-1] = -1.0
            A_ub.append(row1)
            b_ub.append(0.0)
            
            row2 = np.zeros(num_vars)
            row2[j] = -1.0
            row2[worst_idx] = a_jw
            row2[-1] = -1.0
            A_ub.append(row2)
            b_ub.append(0.0)
            
    # 3. sum(w_j) = 1
    A_eq = np.zeros((1, num_vars))
    A_eq[0, :n] = 1.0
    b_eq = [1.0]
    
    # Bornes : w_j in [0, 1], xi >= 0
    bounds = [(0.0, 1.0) for _ in range(n)] + [(0.0, None)]
    
    res = linprog(
        c=c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs"
    )
    
    if not res.success:
        raise ValueError("L'optimisation BWM n'a pas convergé : " + res.message)
        
    weights = res.x[:n]
    xi_star = float(res.x[-1])
    
    return weights, xi_star