import numpy as np
from typing import List

def compute_entropy_weights(decision_matrix: np.ndarray) -> np.ndarray:
    """
    Calcule les poids objectifs par la méthode de l'entropie de Shannon.
    Formule du cours :
      p_ij = d_ij / sum_i(d_ij)
      k = 1 / ln(m)
      E_j = -k * sum_i(p_ij * ln(p_ij))
      w_j = (1 - E_j) / sum_j(1 - E_j)
    """
    m, n = decision_matrix.shape
    if m <= 1:
        return np.ones(n) / n

    # Normalisation par somme de colonne
    col_sums = decision_matrix.sum(axis=0)
    col_sums[col_sums == 0] = 1.0  # Sécurité division par zéro
    p = decision_matrix / col_sums

    # Constante de normalisation
    k = 1.0 / np.log(m)

    # Entropie pour chaque critère
    entropy = np.zeros(n)
    for j in range(n):
        col_p = p[:, j]
        # Convention mathématique : 0 * ln(0) = 0
        valid = col_p > 0
        entropy[j] = -k * np.sum(col_p[valid] * np.log(col_p[valid]))

    # Degré de diversification
    diversification = 1.0 - entropy
    denom = np.sum(diversification)

    if denom == 0:
        return np.ones(n) / n

    weights = diversification / denom
    return weights

def compute_critic_weights(decision_matrix: np.ndarray, criteria_types: List[str]) -> np.ndarray:
    """
    Calcule les poids objectifs par la méthode CRITIC.
    Formule du cours :
      - Normalisation min-max selon bénéfice/coût
      - Corrélation de Pearson rho_jk entre les critères
      - Écart-type sigma_j
      - Indice C_j = sigma_j * sum_k(1 - |rho_jk|)
      - Poids w_j = C_j / sum_j(C_j)
    """
    m, n = decision_matrix.shape
    r = np.zeros_like(decision_matrix, dtype=float)

    # 1. Normalisation Min-Max
    for j in range(n):
        col = decision_matrix[:, j]
        min_val = np.min(col)
        max_val = np.max(col)
        diff = max_val - min_val

        if diff == 0:
            r[:, j] = 1.0
        elif criteria_types[j] == "Bénéfice (+)":
            r[:, j] = (col - min_val) / diff
        else:
            r[:, j] = (max_val - col) / diff

    # 2. Écart-type corrigé (ddof=1)
    stdevs = np.std(r, axis=0, ddof=1) if m > 1 else np.zeros(n)

    # 3. Matrice de corrélation de Pearson
    corr_matrix = np.corrcoef(r, rowvar=False)
    # Remplacement des NaN éventuels (en cas de colonne constante)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # 4. Indice C_j
    C = np.zeros(n)
    for j in range(n):
        independence = np.sum(1.0 - np.abs(corr_matrix[j, :]))
        C[j] = stdevs[j] * independence

    # 5. Poids normalisés
    sum_c = np.sum(C)
    if sum_c == 0:
        return np.ones(n) / n

    weights = C / sum_c
    return weights