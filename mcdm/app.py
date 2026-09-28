import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import linprog

# ==============================================================================
# 1. MOTEUR MATHÉMATIQUE (CORE MCDM)
# ==============================================================================

# --- Table RI pour Saaty (AHP) ---
SAATY_RI = {
    1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.56
}

def compute_ahp(pairwise_matrix: np.ndarray):
    """Calcule les poids AHP et le ratio de cohérence (CR)."""
    n = pairwise_matrix.shape[0]
    col_sums = pairwise_matrix.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    normalized_mat = pairwise_matrix / col_sums
    weights = normalized_mat.mean(axis=1)

    lambda_max = float(np.dot(col_sums, weights))
    if n <= 2:
        return weights, 0.0, 0.0, True

    ci = (lambda_max - n) / (n - 1)
    ri = SAATY_RI.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    return weights, ci, cr, (cr < 0.10)

def compute_bwm(num_criteria: int, best_idx: int, worst_idx: int, bo: np.ndarray, ow: np.ndarray):
    """Résout le modèle de programmation linéaire pour BWM (min xi)."""
    n = num_criteria
    num_vars = n + 1  # [w_1, ..., w_n, xi]

    c = np.zeros(num_vars)
    c[-1] = 1.0

    A_ub = []
    b_ub = []

    # Contraintes |w_B - a_Bj * w_j| <= xi
    for j in range(n):
        if j != best_idx:
            a_bj = float(bo[j])
            r1 = np.zeros(num_vars)
            r1[best_idx] = 1.0
            r1[j] = -a_bj
            r1[-1] = -1.0
            A_ub.append(r1)
            b_ub.append(0.0)

            r2 = np.zeros(num_vars)
            r2[best_idx] = -1.0
            r2[j] = a_bj
            r2[-1] = -1.0
            A_ub.append(r2)
            b_ub.append(0.0)

    # Contraintes |w_j - a_jW * w_W| <= xi
    for j in range(n):
        if j != worst_idx:
            a_jw = float(ow[j])
            r1 = np.zeros(num_vars)
            r1[j] = 1.0
            r1[worst_idx] = -a_jw
            r1[-1] = -1.0
            A_ub.append(r1)
            b_ub.append(0.0)

            r2 = np.zeros(num_vars)
            r2[j] = -1.0
            r2[worst_idx] = a_jw
            r2[-1] = -1.0
            A_ub.append(r2)
            b_ub.append(0.0)

    # Somme des poids = 1
    A_eq = np.zeros((1, num_vars))
    A_eq[0, :n] = 1.0
    b_eq = [1.0]

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
        raise ValueError("Convergence non atteinte pour BWM.")

    return res.x[:n], float(res.x[-1])

def compute_entropy(matrix: np.ndarray):
    """Pondération objective par l'entropie de Shannon."""
    m, n = matrix.shape
    if m <= 1:
        return np.ones(n) / n

    col_sums = matrix.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    p = matrix / col_sums

    k = 1.0 / np.log(m)
    entropy = np.zeros(n)
    for j in range(n):
        col = p[:, j]
        valid = col > 0
        entropy[j] = -k * np.sum(col[valid] * np.log(col[valid]))

    diversification = 1.0 - entropy
    denom = np.sum(diversification)
    return diversification / denom if denom != 0 else np.ones(n) / n

def compute_critic(matrix: np.ndarray, crit_types: list):
    """Pondération objective par CRITIC."""
    m, n = matrix.shape
    r = np.zeros_like(matrix, dtype=float)

    for j in range(n):
        col = matrix[:, j]
        min_v, max_v = np.min(col), np.max(col)
        diff = max_v - min_v
        if diff == 0:
            r[:, j] = 1.0
        elif crit_types[j] == "Bénéfice (+)":
            r[:, j] = (col - min_v) / diff
        else:
            r[:, j] = (max_v - col) / diff

    stdevs = np.std(r, axis=0, ddof=1) if m > 1 else np.zeros(n)
    corr_mat = np.nan_to_num(np.corrcoef(r, rowvar=False), nan=0.0)

    C = np.zeros(n)
    for j in range(n):
        c_diff = 1.0 - np.abs(corr_mat[j, :])
        C[j] = stdevs[j] * np.sum(c_diff)

    sum_c = np.sum(C)
    return C / sum_c if sum_c != 0 else np.ones(n) / n

def rank_wsm(matrix: np.ndarray, weights: np.ndarray, crit_types: list):
    """Classement par la méthode WSM (Somme pondérée)."""
    m, n = matrix.shape
    norm_mat = np.zeros_like(matrix, dtype=float)

    for j in range(n):
        col = matrix[:, j]
        if crit_types[j] == "Bénéfice (+)":
            mx = np.max(col)
            norm_mat[:, j] = col / mx if mx != 0 else 0
        else:
            mn = np.min(col)
            norm_mat[:, j] = mn / col if not np.any(col == 0) else 0

    return np.dot(norm_mat, weights)

def rank_topsis(matrix: np.ndarray, weights: np.ndarray, crit_types: list):
    """Classement par la méthode TOPSIS."""
    m, n = matrix.shape
    sq_sums = np.sqrt(np.sum(matrix ** 2, axis=0))
    sq_sums[sq_sums == 0] = 1.0
    norm_mat = (matrix / sq_sums) * weights

    v_pos = np.zeros(n)
    v_neg = np.zeros(n)

    for j in range(n):
        if crit_types[j] == "Bénéfice (+)":
            v_pos[j] = np.max(norm_mat[:, j])
            v_neg[j] = np.min(norm_mat[:, j])
        else:
            v_pos[j] = np.min(norm_mat[:, j])
            v_neg[j] = np.max(norm_mat[:, j])

    d_pos = np.sqrt(np.sum((norm_mat - v_pos) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((norm_mat - v_neg) ** 2, axis=1))

    denom = d_pos + d_neg
    return np.where(denom == 0, 0, d_neg / denom)

# ==============================================================================
# 2. INTERFACE STREAMLIT
# ==============================================================================

st.set_page_config(page_title="MCDM Suite - Aide à la Décision", layout="wide")

st.title("Système d'Aide à la Décision Multicritère (MCDM)")
st.markdown("Pondération des critères (AHP, BWM, Entropie, CRITIC) et classement des alternatives (WSM, TOPSIS).")

# Initialisation de l'état
if "weights" not in st.session_state:
    st.session_state.weights = None
if "criteria_names" not in st.session_state:
    st.session_state.criteria_names = ["Coût", "Qualité", "Délai"]
if "alt_names" not in st.session_state:
    st.session_state.alt_names = ["Alternative 1", "Alternative 2", "Alternative 3"]

# --- SECTION 1 : CONFIGURATION ET MATRICE DE DÉCISION ---
st.header("1. Paramétrage des critères et alternatives")

col_cfg1, col_cfg2 = st.columns(2)
with col_cfg1:
    crit_input = st.text_input("Critères (séparés par virgule) :", value=", ".join(st.session_state.criteria_names))
    criteria = [c.strip() for c in crit_input.split(",") if c.strip()]
with col_cfg2:
    alt_input = st.text_input("Alternatives (séparées par virgule) :", value=", ".join(st.session_state.alt_names))
    alternatives = [a.strip() for a in alt_input.split(",") if a.strip()]

num_crit = len(criteria)
num_alt = len(alternatives)

if num_crit < 2 or num_alt < 2:
    st.warning("Veuillez renseigner au moins 2 critères et 2 alternatives.")
    st.stop()

st.write("**Typologie des critères :**")
crit_types = []
type_cols = st.columns(num_crit)
for idx, col in enumerate(type_cols):
    with col:
        t = st.selectbox(f"{criteria[idx]}", ["Bénéfice (+)", "Coût (-)"], key=f"type_{idx}")
        crit_types.append(t)

st.write("**Matrice de décision :**")
init_data = np.ones((num_alt, num_crit)) * 10.0
df_decision = pd.DataFrame(init_data, index=alternatives, columns=criteria)
edited_decision_df = st.data_editor(df_decision, use_container_width=True)
decision_matrix = edited_decision_df.to_numpy()

st.markdown("---")

# --- SECTION 2 : PONDÉRATION DES CRITÈRES ---
st.header("2. Détermination des poids des critères")

family = st.radio(
    "Famille de méthode :",
    ["Méthodes subjectives (Jugements d'experts)", "Méthodes objectives (Basées sur les données)"]
)

scale_saaty = {
    1: "1 - Égale importance",
    2: "2 - Égale à modérée",
    3: "3 - Modérément plus important",
    4: "4 - Modérée à forte",
    5: "5 - Fortement plus important",
    6: "6 - Forte à très forte",
    7: "7 - Très fortement plus important",
    8: "8 - Très forte à extrême",
    9: "9 - Extrêmement plus important"
}

if family == "Méthodes subjectives (Jugements d'experts)":
    sub_method = st.selectbox("Méthode subjective :", ["AHP (Analytic Hierarchy Process)", "BWM (Best-Worst Method)"])

    if sub_method.startswith("AHP"):
        st.subheader("Questionnaire AHP (Comparaisons par paires)")
        st.info("Échelle de Saaty : 1 = égalité, 3 = modéré, 5 = fort, 7 = très fort, 9 = extrême.")

        pairwise_mat = np.ones((num_crit, num_crit))
        for i in range(num_crit):
            for j in range(i + 1, num_crit):
                c_pref, c_score = st.columns([2, 1])
                with c_pref:
                    fav = st.radio(
                        f"Préférence entre **{criteria[i]}** et **{criteria[j]}** :",
                        [criteria[i], criteria[j]],
                        key=f"ahp_fav_{i}_{j}"
                    )
                with c_score:
                    val = st.selectbox(
                        "Degré d'importance :",
                        options=list(scale_saaty.keys()),
                        format_func=lambda x: scale_saaty[x],
                        key=f"ahp_val_{i}_{j}"
                    )

                if fav == criteria[i]:
                    pairwise_mat[i, j] = val
                    pairwise_mat[j, i] = 1.0 / val
                else:
                    pairwise_mat[i, j] = 1.0 / val
                    pairwise_mat[j, i] = val

        if st.button("Calculer les poids AHP"):
            w, ci, cr, is_valid = compute_ahp(pairwise_mat)
            st.session_state.weights = w
            m1, m2, m3 = st.columns(3)
            m1.metric("CI (Indice de cohérence)", f"{ci:.4f}")
            m2.metric("CR (Ratio de cohérence)", f"{cr:.4f}")
            m3.metric("Test de cohérence", "Validé (CR < 0.1)" if is_valid else "Incohérent (CR ≥ 0.1)")
            if not is_valid:
                st.error("Le ratio de cohérence dépasse 10%. Veuillez ajuster vos jugements.")

    elif sub_method.startswith("BWM"):
        st.subheader("Questionnaire BWM (Best-Worst Method)")
        col_b, col_w = st.columns(2)
        with col_b:
            b_crit = st.selectbox("Critère le plus important (Best) :", criteria, index=0)
        with col_w:
            w_crit = st.selectbox("Critère le moins important (Worst) :", criteria, index=num_crit - 1)

        if b_crit == w_crit:
            st.error("Le meilleur critère et le pire critère doivent être différents.")
        else:
            b_idx = criteria.index(b_crit)
            w_idx = criteria.index(w_crit)

            st.write(f"**Vecteur Best-to-Others (BO) : Importance de '{b_crit}' par rapport aux autres critères**")
            bo_vec = np.ones(num_crit)
            for j in range(num_crit):
                if j == b_idx:
                    bo_vec[j] = 1.0
                else:
                    bo_vec[j] = st.slider(
                        f"Importance de [{b_crit}] par rapport à [{criteria[j]}] :",
                        1, 9, 3, key=f"bwm_bo_{j}"
                    )

            st.write(f"**Vecteur Others-to-Worst (OW) : Importance de chaque critère par rapport à '{w_crit}'**")
            ow_vec = np.ones(num_crit)
            for j in range(num_crit):
                if j == w_idx:
                    ow_vec[j] = 1.0
                else:
                    ow_vec[j] = st.slider(
                        f"Importance de [{criteria[j]}] par rapport à [{w_crit}] :",
                        1, 9, 3, key=f"bwm_ow_{j}"
                    )

            if st.button("Calculer les poids BWM"):
                try:
                    w, xi = compute_bwm(num_crit, b_idx, w_idx, bo_vec, ow_vec)
                    st.session_state.weights = w
                    st.metric("Indice d'incohérence ξ*", f"{xi:.4f}")
                except Exception as ex:
                    st.error(f"Erreur d'optimisation : {ex}")

else:
    obj_method = st.selectbox("Méthode objective :", ["Entropie de Shannon", "CRITIC"])
    st.info("Les méthodes objectives dérivent les poids directement de la matrice de décision.")

    if st.button(f"Calculer les poids par {obj_method}"):
        if obj_method.startswith("Entropie"):
            st.session_state.weights = compute_entropy(decision_matrix)
        else:
            st.session_state.weights = compute_critic(decision_matrix, crit_types)

# Affichage des poids
if st.session_state.weights is not None:
    df_w = pd.DataFrame({"Critère": criteria, "Poids": st.session_state.weights})
    st.write("### Poids calculés")
    c_chart, c_tbl = st.columns([2, 1])
    with c_chart:
        fig = px.bar(df_w, x="Critère", y="Poids", text_auto=".4f", title="Distribution des poids")
        st.plotly_chart(fig, use_container_width=True)
    with c_tbl:
        st.dataframe(df_w.style.format({"Poids": "{:.4f}"}), use_container_width=True)

    # --- SECTION 3 : CLASSEMENT DES ALTERNATIVES ---
    st.markdown("---")
    st.header("3. Classement des alternatives")

    rank_alg = st.selectbox("Algorithme de classement :", ["TOPSIS", "WSM (Somme pondérée)"])

    if st.button("Lancer le classement"):
        w = st.session_state.weights
        if rank_alg == "TOPSIS":
            scores = rank_topsis(decision_matrix, w, crit_types)
            metric_label = "Proximité relative (RC*)"
        else:
            scores = rank_wsm(decision_matrix, w, crit_types)
            metric_label = "Score WSM"

        df_res = pd.DataFrame({
            "Alternative": alternatives,
            metric_label: scores
        }).sort_values(by=metric_label, ascending=False).reset_index(drop=True)

        df_res["Rang"] = df_res.index + 1

        st.subheader("Classement final")
        st.dataframe(
            df_res[["Rang", "Alternative", metric_label]].style.format({metric_label: "{:.4f}"}),
            use_container_width=True
        )

        fig_rank = px.bar(
            df_res,
            x="Alternative",
            y=metric_label,
            color=metric_label,
            text_auto=".4f",
            title=f"Classement des alternatives ({rank_alg})"
        )
        st.plotly_chart(fig_rank, use_container_width=True)
