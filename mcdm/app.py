import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import linprog

# ==============================================================================
# CONFIGURATION DE LA PAGE & STYLES CSS
# ==============================================================================
st.set_page_config(page_title="MCDM", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    .grid-header {
        font-weight: 600;
        font-size: 0.9rem;
        color: #94a3b8;
        padding-bottom: 8px;
        border-bottom: 1px solid #334155;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MOTEUR DE CALCUL
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
        raise ValueError("Convergence non atteinte pour BWM.")
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
# INTERFACE UTILISATEUR
# ==============================================================================

st.title("Système MCDM")
st.caption("Aide à la décision multicritère (AHP, BWM, TOPSIS, WSM)")

# --- 1. CONFIGURATION ---
with st.container(border=True):
    st.subheader("1. Paramétrage")
    col1, col2 = st.columns(2)
    with col1:
        crit_raw = st.text_input("Critères (séparés par virgules) :", "Coût, Confort, Sécurité")
        criteria = [c.strip() for c in crit_raw.split(",") if c.strip()]
    with col2:
        alt_raw = st.text_input("Alternatives (séparées par virgules) :", "Voiture 1, Voiture 2, Voiture 3")
        alternatives = [a.strip() for a in alt_raw.split(",") if a.strip()]

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        method_weights = st.selectbox("Méthode de pondération des critères :", ["AHP", "BWM"])
    with col_m2:
        method_ranking = st.selectbox("Méthode de classement des alternatives :", ["AHP", "TOPSIS", "WSM"])

num_crit = len(criteria)
num_alt = len(alternatives)

if num_crit < 2 or num_alt < 2:
    st.warning("Veuillez renseigner au moins 2 critères et 2 alternatives.")
    st.stop()

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

crit_weights = None

# --- 2. PONDÉRATION DES CRITÈRES ---
with st.container(border=True):
    st.subheader(f"2. Pondération des critères ({method_weights})")

    if method_weights == "AHP":
        # En-têtes de colonnes
        h1, h2, h3 = st.columns([3, 4, 3])
        h1.markdown("<div class='grid-header'>Paire de critères</div>", unsafe_allow_html=True)
        h2.markdown("<div class='grid-header'>Critère dominant</div>", unsafe_allow_html=True)
        h3.markdown("<div class='grid-header'>Intensité (1 à 9)</div>", unsafe_allow_html=True)

        crit_matrix = np.ones((num_crit, num_crit))

        for i in range(num_crit):
            for j in range(i + 1, num_crit):
                c_pair, c_fav, c_deg = st.columns([3, 4, 3])
                with c_pair:
                    st.write(f"**{criteria[i]}** vs **{criteria[j]}**")
                with c_fav:
                    fav = st.radio(
                        f"fav_{i}_{j}",
                        [criteria[i], criteria[j]],
                        horizontal=True,
                        label_visibility="collapsed",
                        key=f"ahp_fav_{i}_{j}"
                    )
                with c_deg:
                    val = st.selectbox(
                        f"val_{i}_{j}",
                        options=list(saaty_scale.keys()),
                        format_func=lambda x: saaty_scale[x],
                        label_visibility="collapsed",
                        key=f"ahp_val_{i}_{j}"
                    )

                if fav == criteria[i]:
                    crit_matrix[i, j] = val
                    crit_matrix[j, i] = 1.0 / val
                else:
                    crit_matrix[i, j] = 1.0 / val
                    crit_matrix[j, i] = val

        weights, ci, cr, is_consistent = compute_ahp(crit_matrix)
        crit_weights = weights

        st.write("")
        with st.container(border=True):
            m1, m2, m3 = st.columns(3)
            m1.metric("CI (Indice de cohérence)", f"{ci:.4f}")
            m2.metric("CR (Ratio de cohérence)", f"{cr:.4f}")
            m3.metric("Cohérence globale", "Validée (CR < 0.1)" if is_consistent else "Incohérente (CR ≥ 0.1)")

    elif method_weights == "BWM":
        c_b, c_w = st.columns(2)
        with c_b:
            best_c = st.selectbox("Critère le plus important (Best) :", criteria, index=0)
        with c_w:
            worst_c = st.selectbox("Critère le moins important (Worst) :", criteria, index=num_crit - 1)

        if best_c == worst_c:
            st.error("Le meilleur critère et le pire critère doivent être distincts.")
            st.stop()

        b_idx = criteria.index(best_c)
        w_idx = criteria.index(worst_c)

        st.write(f"**Comparaisons par rapport au meilleur critère ({best_c}) :**")
        bo_vec = np.ones(num_crit)
        for j in range(num_crit):
            if j != b_idx:
                c1, c2 = st.columns([3, 7])
                c1.write(f"**{best_c}** vs **{criteria[j]}**")
                bo_vec[j] = c2.slider(f"bo_{j}", 1, 9, 3, label_visibility="collapsed", key=f"bo_{j}")

        st.write(f"**Comparaisons par rapport au pire critère ({worst_c}) :**")
        ow_vec = np.ones(num_crit)
        for j in range(num_crit):
            if j != w_idx:
                c1, c2 = st.columns([3, 7])
                c1.write(f"**{criteria[j]}** vs **{worst_c}**")
                ow_vec[j] = c2.slider(f"ow_{j}", 1, 9, 3, label_visibility="collapsed", key=f"ow_{j}")

        weights, xi = compute_bwm(num_crit, b_idx, w_idx, bo_vec, ow_vec)
        crit_weights = weights

        with st.container(border=True):
            st.metric("Indice d'incohérence optimal (ξ*)", f"{xi:.4f}")

    # Restitution graphique des poids
    st.write("")
    c_chart, c_tbl = st.columns([3, 2])
    df_w = pd.DataFrame({"Critère": criteria, "Poids": crit_weights})
    with c_chart:
        fig_w = px.bar(df_w, x="Critère", y="Poids", text_auto=".3f", title="Distribution des poids")
        fig_w.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_w, use_container_width=True)
    with c_tbl:
        st.write("**Valeurs exactes des poids :**")
        st.dataframe(
            df_w.style.format({"Poids": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
            height=240
        )

# --- 3. ÉVALUATION ET CLASSEMENT ---
with st.container(border=True):
    st.subheader(f"3. Évaluation des alternatives ({method_ranking})")

    if method_ranking == "AHP":
        st.write("Comparaison deux à deux des alternatives pour chaque critère :")
        local_priorities = np.zeros((num_alt, num_crit))

        for k, crit_name in enumerate(criteria):
            with st.expander(f"Critère : {crit_name}", expanded=(k == 0)):
                h1, h2, h3 = st.columns([3, 4, 3])
                h1.markdown("<div class='grid-header'>Paire d'alternatives</div>", unsafe_allow_html=True)
                h2.markdown("<div class='grid-header'>Option préférée</div>", unsafe_allow_html=True)
                h3.markdown("<div class='grid-header'>Intensité (1 à 9)</div>", unsafe_allow_html=True)

                mat_alt = np.ones((num_alt, num_alt))
                for i in range(num_alt):
                    for j in range(i + 1, num_alt):
                        c_p, c_f, c_d = st.columns([3, 4, 3])
                        c_p.write(f"**{alternatives[i]}** vs **{alternatives[j]}**")
                        fav_a = c_f.radio(
                            f"fav_a_{k}_{i}_{j}",
                            [alternatives[i], alternatives[j]],
                            horizontal=True,
                            label_visibility="collapsed",
                            key=f"alt_fav_{k}_{i}_{j}"
                        )
                        val_a = c_d.selectbox(
                            f"val_a_{k}_{i}_{j}",
                            options=list(saaty_scale.keys()),
                            format_func=lambda x: saaty_scale[x],
                            label_visibility="collapsed",
                            key=f"alt_val_{k}_{i}_{j}"
                        )

                        if fav_a == alternatives[i]:
                            mat_alt[i, j] = val_a
                            mat_alt[j, i] = 1.0 / val_a
                        else:
                            mat_alt[i, j] = 1.0 / val_a
                            mat_alt[j, i] = val_a

                p_loc, _, cr_a, _ = compute_ahp(mat_alt)
                local_priorities[:, k] = p_loc
                st.caption(f"Ratio de cohérence locale (CR) = {cr_a:.4f}")

        if st.button("Calculer le classement final (AHP)", type="primary"):
            final_scores = np.dot(local_priorities, crit_weights)
            df_res = pd.DataFrame({"Alternative": alternatives, "Score AHP": final_scores})
            df_res = df_res.sort_values(by="Score AHP", ascending=False).reset_index(drop=True)
            df_res["Rang"] = df_res.index + 1

            st.write("---")
            c_r1, c_r2 = st.columns([3, 2])
            with c_r1:
                fig_r = px.bar(df_res, x="Alternative", y="Score AHP", text_auto=".4f", title="Classement AHP", color="Score AHP")
                fig_r.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_r, use_container_width=True)
            with c_r2:
                st.write("**Résultats finaux :**")
                st.dataframe(df_res[["Rang", "Alternative", "Score AHP"]].style.format({"Score AHP": "{:.4f}"}), use_container_width=True, hide_index=True)

    else:
        # TOPSIS / WSM
        st.write("**Définition des sens d'optimisation :**")
        crit_types = []
        t_cols = st.columns(num_crit)
        for idx, c in enumerate(t_cols):
            with c:
                t = st.selectbox(f"{criteria[idx]} :", ["Bénéfice (+)", "Coût (-)"], key=f"t_{idx}")
                crit_types.append(t)

        st.write("**Notation des alternatives (Échelle de performance 1 à 9) :**")
        score_matrix = np.zeros((num_alt, num_crit))

        for i, alt in enumerate(alternatives):
            with st.expander(f"Alternative : {alt}", expanded=True):
                eval_cols = st.columns(num_crit)
                for j, crit in enumerate(criteria):
                    with eval_cols[j]:
                        score_matrix[i, j] = st.slider(f"{crit} :", 1, 9, 5, key=f"eval_{i}_{j}")

        if st.button(f"Calculer le classement ({method_ranking})", type="primary"):
            if method_ranking == "TOPSIS":
                scores = run_topsis(score_matrix, crit_weights, crit_types)
                score_col = "Score TOPSIS (RC*)"
            else:
                scores = run_wsm(score_matrix, crit_weights, crit_types)
                score_col = "Score WSM"

            df_res = pd.DataFrame({"Alternative": alternatives, score_col: scores})
            df_res = df_res.sort_values(by=score_col, ascending=False).reset_index(drop=True)
            df_res["Rang"] = df_res.index + 1

            st.write("---")
            c_r1, c_r2 = st.columns([3, 2])
            with c_r1:
                fig_r = px.bar(df_res, x="Alternative", y=score_col, text_auto=".4f", title=f"Classement ({method_ranking})", color=score_col)
                fig_r.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_r, use_container_width=True)
            with c_r2:
                st.write("**Résultats finaux :**")
                st.dataframe(df_res[["Rang", "Alternative", score_col]].style.format({score_col: "{:.4f}"}), use_container_width=True, hide_index=True)
