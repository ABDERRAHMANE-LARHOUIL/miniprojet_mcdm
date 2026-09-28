import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import linprog

# ==============================================================================
# 1. CALCULS MCDM
# ==============================================================================

SAATY_RI = {
    1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.56
}

def compute_ahp(matrix: np.ndarray):
    n = matrix.shape[0]
    col_sums = matrix.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    norm_mat = matrix / col_sums
    weights = norm_mat.mean(axis=1)

    lambda_max = float(np.dot(col_sums, weights))
    if n <= 2:
        return weights, 0.0, 0.0, True

    ci = (lambda_max - n) / (n - 1)
    ri = SAATY_RI.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    return weights, ci, cr, (cr < 0.10)

def compute_bwm(num_crit: int, best_idx: int, worst_idx: int, bo: np.ndarray, ow: np.ndarray):
    num_vars = num_crit + 1
    c = np.zeros(num_vars)
    c[-1] = 1.0

    A_ub, b_ub = [], []
    for j in range(num_crit):
        if j != best_idx:
            a_bj = float(bo[j])
            r1, r2 = np.zeros(num_vars), np.zeros(num_vars)
            r1[best_idx], r1[j], r1[-1] = 1.0, -a_bj, -1.0
            r2[best_idx], r2[j], r2[-1] = -1.0, a_bj, -1.0
            A_ub.extend([r1, r2])
            b_ub.extend([0.0, 0.0])

        if j != worst_idx:
            a_jw = float(ow[j])
            r1, r2 = np.zeros(num_vars), np.zeros(num_vars)
            r1[j], r1[worst_idx], r1[-1] = 1.0, -a_jw, -1.0
            r2[j], r2[worst_idx], r2[-1] = -1.0, a_jw, -1.0
            A_ub.extend([r1, r2])
            b_ub.extend([0.0, 0.0])

    A_eq = np.zeros((1, num_vars))
    A_eq[0, :num_crit] = 1.0
    b_eq = [1.0]
    bounds = [(0.0, 1.0) for _ in range(num_crit)] + [(0.0, None)]

    res = linprog(c=c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise ValueError("Erreur de convergence BWM")
    return res.x[:num_crit], float(res.x[-1])

def run_topsis(matrix: np.ndarray, weights: np.ndarray, crit_types: list):
    n = matrix.shape[1]
    sq_sums = np.sqrt(np.sum(matrix ** 2, axis=0))
    sq_sums[sq_sums == 0] = 1.0
    norm_mat = (matrix / sq_sums) * weights

    v_pos, v_neg = np.zeros(n), np.zeros(n)
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

def run_wsm(matrix: np.ndarray, weights: np.ndarray, crit_types: list):
    n = matrix.shape[1]
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

# ==============================================================================
# 2. INTERFACE UTILISATEUR
# ==============================================================================

st.set_page_config(page_title="MCDM", layout="wide")
st.title("Aide à la Décision Multicritère (MCDM)")

# --- 1. CONFIGURATION ---
st.header("1. Critères et Alternatives")

c_crit, c_alt = st.columns(2)
with c_crit:
    crit_input = st.text_input("Critères (séparés par virgule) :", "Coût, Confort, Sécurité")
    criteria = [c.strip() for c in crit_input.split(",") if c.strip()]
with c_alt:
    alt_input = st.text_input("Alternatives (séparées par virgule) :", "Option 1, Option 2, Option 3")
    alternatives = [a.strip() for a in alt_input.split(",") if a.strip()]

num_crit = len(criteria)
num_alt = len(alternatives)

if num_crit < 2 or num_alt < 2:
    st.warning("Il faut au minimum 2 critères et 2 alternatives.")
    st.stop()

col_m1, col_m2 = st.columns(2)
with col_m1:
    method_weights = st.selectbox("Méthode de pondération :", ["AHP", "BWM"])
with col_m2:
    method_ranking = st.selectbox("Méthode de classement :", ["AHP", "TOPSIS", "WSM"])

st.markdown("---")

# Échelle Saaty 1 à 9
saaty_scale = {
    1: "1 - Égal",
    2: "2 - Intermédiaire",
    3: "3 - Modéré",
    4: "4 - Intermédiaire",
    5: "5 - Fort",
    6: "6 - Intermédiaire",
    7: "7 - Très fort",
    8: "8 - Intermédiaire",
    9: "9 - Extrême"
}

# --- 2. PONDÉRATION DES CRITÈRES ---
st.header(f"2. Pondération des critères ({method_weights})")

crit_weights = None

if method_weights == "AHP":
    crit_matrix = np.ones((num_crit, num_crit))
    for i in range(num_crit):
        for j in range(i + 1, num_crit):
            c_sel, c_val = st.columns([2, 1])
            with c_sel:
                fav = st.radio(
                    f"Critère préféré :",
                    [criteria[i], criteria[j]],
                    key=f"w_crit_{i}_{j}"
                )
            with c_val:
                val = st.selectbox(
                    "Valeur (1-9) :",
                    options=list(saaty_scale.keys()),
                    format_func=lambda x: saaty_scale[x],
                    key=f"w_val_{i}_{j}"
                )

            if fav == criteria[i]:
                crit_matrix[i, j] = val
                crit_matrix[j, i] = 1.0 / val
            else:
                crit_matrix[i, j] = 1.0 / val
                crit_matrix[j, i] = val

    weights, ci, cr, is_consistent = compute_ahp(crit_matrix)
    crit_weights = weights

    res1, res2, res3 = st.columns(3)
    res1.metric("CI", f"{ci:.4f}")
    res2.metric("CR", f"{cr:.4f}")
    res3.metric("Cohérence", "Validée (CR < 0.1)" if is_consistent else "Incohérente (CR ≥ 0.1)")

elif method_weights == "BWM":
    col_b, col_w = st.columns(2)
    with col_b:
        best_crit = st.selectbox("Meilleur critère (Best) :", criteria, index=0)
    with col_w:
        worst_crit = st.selectbox("Pire critère (Worst) :", criteria, index=num_crit - 1)

    if best_crit == worst_crit:
        st.error("Le meilleur et le pire critère doivent être différents.")
        st.stop()

    b_idx = criteria.index(best_crit)
    w_idx = criteria.index(worst_crit)

    st.write(f"**Comparaison Best-to-Others (BO) : importance de [{best_crit}] par rapport aux autres**")
    bo_vec = np.ones(num_crit)
    for j in range(num_crit):
        if j != b_idx:
            bo_vec[j] = st.slider(f"[{best_crit}] vs [{criteria[j]}] :", 1, 9, 3, key=f"bo_{j}")

    st.write(f"**Comparaison Others-to-Worst (OW) : importance des critères par rapport à [{worst_crit}]**")
    ow_vec = np.ones(num_crit)
    for j in range(num_crit):
        if j != w_idx:
            ow_vec[j] = st.slider(f"[{criteria[j]}] vs [{worst_crit}] :", 1, 9, 3, key=f"ow_{j}")

    weights, xi = compute_bwm(num_crit, b_idx, w_idx, bo_vec, ow_vec)
    crit_weights = weights
    st.metric("Indice d'incohérence ξ*", f"{xi:.4f}")

# Tableau des poids calculés
df_w = pd.DataFrame({"Critère": criteria, "Poids": crit_weights})
st.dataframe(df_w.set_index("Critère").T.style.format("{:.4f}"), use_container_width=True)

st.markdown("---")

# --- 3. QUESTIONNAIRE ALTERNATIVES ---
st.header(f"3. Évaluation des alternatives ({method_ranking})")

if method_ranking == "AHP":
    local_priorities = np.zeros((num_alt, num_crit))

    for k, crit_name in enumerate(criteria):
        with st.expander(f"Critère : {crit_name}", expanded=(k == 0)):
            mat_alt = np.ones((num_alt, num_alt))
            for i in range(num_alt):
                for j in range(i + 1, num_alt):
                    col_p, col_v = st.columns([2, 1])
                    with col_p:
                        fav_alt = st.radio(
                            f"Meilleure option :",
                            [alternatives[i], alternatives[j]],
                            key=f"alt_{k}_{i}_{j}"
                        )
                    with col_v:
                        deg_alt = st.selectbox(
                            "Valeur (1-9) :",
                            options=list(saaty_scale.keys()),
                            format_func=lambda x: saaty_scale[x],
                            key=f"alt_deg_{k}_{i}_{j}"
                        )

                    if fav_alt == alternatives[i]:
                        mat_alt[i, j] = deg_alt
                        mat_alt[j, i] = 1.0 / deg_alt
                    else:
                        mat_alt[i, j] = 1.0 / deg_alt
                        mat_alt[j, i] = deg_alt

            p_loc, _, cr_a, _ = compute_ahp(mat_alt)
            local_priorities[:, k] = p_loc
            st.caption(f"CR ({crit_name}) = {cr_a:.4f}")

    if st.button("Calculer le classement final"):
        final_scores = np.dot(local_priorities, crit_weights)
        df_res = pd.DataFrame({"Alternative": alternatives, "Score AHP": final_scores})
        df_res = df_res.sort_values(by="Score AHP", ascending=False).reset_index(drop=True)
        df_res["Rang"] = df_res.index + 1

        st.subheader("Classement Final")
        st.dataframe(df_res[["Rang", "Alternative", "Score AHP"]].style.format({"Score AHP": "{:.4f}"}), use_container_width=True)

        fig = px.bar(df_res, x="Alternative", y="Score AHP", color="Alternative", text_auto=".4f")
        st.plotly_chart(fig, use_container_width=True)

else:
    # TOPSIS ou WSM
    st.write("**Type de critère :**")
    crit_types = []
    type_cols = st.columns(num_crit)
    for idx, c in enumerate(type_cols):
        with c:
            t = st.selectbox(criteria[idx], ["Bénéfice (+)", "Coût (-)"], key=f"t_{idx}")
            crit_types.append(t)

    score_matrix = np.zeros((num_alt, num_crit))

    for i, alt in enumerate(alternatives):
        with st.expander(f"Alternative : {alt}", expanded=True):
            eval_cols = st.columns(num_crit)
            for j, crit in enumerate(criteria):
                with eval_cols[j]:
                    score_matrix[i, j] = st.slider(f"{crit} (1-9) :", 1, 9, 5, key=f"eval_{i}_{j}")

    if st.button("Calculer le classement final"):
        if method_ranking == "TOPSIS":
            scores = run_topsis(score_matrix, crit_weights, crit_types)
            col_name = "Score TOPSIS (RC*)"
        else:
            scores = run_wsm(score_matrix, crit_weights, crit_types)
            col_name = "Score WSM"

        df_res = pd.DataFrame({"Alternative": alternatives, col_name: scores})
        df_res = df_res.sort_values(by=col_name, ascending=False).reset_index(drop=True)
        df_res["Rang"] = df_res.index + 1

        st.subheader("Classement Final")
        st.dataframe(df_res[["Rang", "Alternative", col_name]].style.format({col_name: "{:.4f}"}), use_container_width=True)

        fig = px.bar(df_res, x="Alternative", y=col_name, color="Alternative", text_auto=".4f")
        st.plotly_chart(fig, use_container_width=True)
