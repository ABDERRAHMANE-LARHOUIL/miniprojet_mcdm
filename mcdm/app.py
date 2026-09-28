import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import linprog

# ==============================================================================
# 1. MOTEUR DE CALCUL (MCDM)
# ==============================================================================

SAATY_RI = {
    1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.56
}

def compute_ahp_vector(matrix: np.ndarray):
    """Calcule le vecteur de priorités et le ratio de cohérence (CR)."""
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

def compute_bwm_weights(num_crit: int, best_idx: int, worst_idx: int, bo: np.ndarray, ow: np.ndarray):
    """Optimisation linéaire BWM (min xi)."""
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
        raise ValueError("Optimisation BWM impossible : " + res.message)
    return res.x[:num_crit], float(res.x[-1])

def run_topsis_calc(decision_matrix: np.ndarray, weights: np.ndarray, crit_types: list):
    """Algorithme TOPSIS appliqué à la matrice générée par le questionnaire."""
    sq_sums = np.sqrt(np.sum(decision_matrix ** 2, axis=0))
    sq_sums[sq_sums == 0] = 1.0
    norm_mat = (decision_matrix / sq_sums) * weights
    n = decision_matrix.shape[1]

    v_pos, v_neg = np.zeros(n), np.zeros(n)
    for j in range(n):
        if crit_types[j] == "Bénéfice (Plus c'est élevé, mieux c'est)":
            v_pos[j] = np.max(norm_mat[:, j])
            v_neg[j] = np.min(norm_mat[:, j])
        else:
            v_pos[j] = np.min(norm_mat[:, j])
            v_neg[j] = np.max(norm_mat[:, j])

    d_pos = np.sqrt(np.sum((norm_mat - v_pos) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((norm_mat - v_neg) ** 2, axis=1))
    denom = d_pos + d_neg
    return np.where(denom == 0, 0, d_neg / denom)

def run_wsm_calc(decision_matrix: np.ndarray, weights: np.ndarray, crit_types: list):
    """Algorithme WSM appliqué à la matrice générée par le questionnaire."""
    m, n = decision_matrix.shape
    norm_mat = np.zeros_like(decision_matrix, dtype=float)
    for j in range(n):
        col = decision_matrix[:, j]
        if crit_types[j] == "Bénéfice (Plus c'est élevé, mieux c'est)":
            mx = np.max(col)
            norm_mat[:, j] = col / mx if mx != 0 else 0
        else:
            mn = np.min(col)
            norm_mat[:, j] = mn / col if not np.any(col == 0) else 0
    return np.dot(norm_mat, weights)

# ==============================================================================
# 2. APPLICATION STREAMLIT (100% QUESTIONNAIRE)
# ==============================================================================

st.set_page_config(page_title="Système Expert MCDM", layout="wide")

st.title("Aide à la Décision Multicritère (Approche Guidée par Questionnaire)")
st.caption("Évaluation subjective complète sans manipulation manuelle de matrices ou de tableaux numériques.")

# --- ÉTAPE 1 : IDENTIFICATION DU PROBLÈME ---
st.header("1. Cadrage du problème")

c1, c2 = st.columns(2)
with c1:
    crit_text = st.text_input("Critères d'évaluation (séparés par des virgules) :", "Coût, Confort, Sécurité")
    criteria = [c.strip() for c in crit_text.split(",") if c.strip()]
with c2:
    alt_text = st.text_input("Alternatives / Options candidates (séparées par des virgules) :", "Voiture 1, Voiture 2, Voiture 3")
    alternatives = [a.strip() for a in alt_text.split(",") if a.strip()]

num_crit = len(criteria)
num_alt = len(alternatives)

if num_crit < 2 or num_alt < 2:
    st.warning("Veuillez renseigner au moins 2 critères et 2 alternatives.")
    st.stop()

eval_strategy = st.radio(
    "Mode d'évaluation souhaité :",
    [
        "AHP Hiérarchique Complet (Comparaisons par paires des critères et des alternatives)",
        "Questionnaire Qualitatif Guidé (Pondération subjective + Évaluation par échelle linguistique)"
    ]
)

st.markdown("---")

# --- ÉTAPE 2 : QUESTIONNAIRE SUR LES CRITÈRES ---
st.header("2. Questionnaire de pondération des critères")

saaty_options = {
    1: "1 - Importance égale",
    3: "3 - Modérément plus important",
    5: "5 - Fortement plus important",
    7: "7 - Très fortement plus important",
    9: "9 - Extrêmement plus important"
}

crit_weights = None

if eval_strategy.startswith("AHP Hiérarchique"):
    st.subheader("Comparaison deux à deux des critères")
    crit_matrix = np.ones((num_crit, num_crit))

    for i in range(num_crit):
        for j in range(i + 1, num_crit):
            c_p, c_deg = st.columns([2, 2])
            with c_p:
                fav = st.radio(
                    f"Lequel est le plus important selon vous ?",
                    [criteria[i], criteria[j]],
                    key=f"crit_pref_{i}_{j}"
                )
            with c_deg:
                deg = st.selectbox(
                    f"Avec quel degré de supériorité ?",
                    options=list(saaty_options.keys()),
                    format_func=lambda x: saaty_options[x],
                    key=f"crit_deg_{i}_{j}"
                )

            if fav == criteria[i]:
                crit_matrix[i, j] = deg
                crit_matrix[j, i] = 1.0 / deg
            else:
                crit_matrix[i, j] = 1.0 / deg
                crit_matrix[j, i] = deg

    weights_c, ci, cr, ok = compute_ahp_vector(crit_matrix)
    crit_weights = weights_c

    st.write(f"**Indice de cohérence (CI)** : `{ci:.4f}` | **Ratio de cohérence (CR)** : `{cr:.4f}`")
    if not ok:
        st.error("Attention : vos réponses sur les critères manquent de transitivité (CR ≥ 10%).")
    else:
        st.success("Jugements sur les critères cohérents (CR < 10%).")

else:
    sub_method = st.selectbox("Méthode de pondération des critères :", ["AHP (Comparaisons par paires)", "BWM (Best-Worst Method)"])
    
    if sub_method.startswith("AHP"):
        crit_matrix = np.ones((num_crit, num_crit))
        for i in range(num_crit):
            for j in range(i + 1, num_crit):
                cp1, cp2 = st.columns([2, 2])
                with cp1:
                    fav = st.radio(f"Critère prioritaire :", [criteria[i], criteria[j]], key=f"q_c_{i}_{j}")
                with cp2:
                    deg = st.selectbox("Importance :", list(saaty_options.keys()), format_func=lambda x: saaty_options[x], key=f"deg_c_{i}_{j}")
                if fav == criteria[i]:
                    crit_matrix[i, j] = deg
                    crit_matrix[j, i] = 1.0 / deg
                else:
                    crit_matrix[i, j] = 1.0 / deg
                    crit_matrix[j, i] = deg

        weights_c, _, cr, _ = compute_ahp_vector(crit_matrix)
        crit_weights = weights_c
    else:
        st.write("Désignez le critère le plus important et le moins important :")
        cb, cw = st.columns(2)
        with cb:
            best_c = st.selectbox("Critère le plus déterminant (Best) :", criteria, index=0)
        with cw:
            worst_c = st.selectbox("Critère le moins déterminant (Worst) :", criteria, index=num_crit - 1)

        if best_c == worst_c:
            st.error("Le meilleur et le pire critère doivent être différents.")
            st.stop()

        b_idx, w_idx = criteria.index(best_c), criteria.index(worst_c)

        bo_vec = np.ones(num_crit)
        ow_vec = np.ones(num_crit)

        st.write(f"**À quel point [{best_c}] est-il plus prioritaire que chacun des autres critères ?**")
        for j in range(num_crit):
            if j != b_idx:
                bo_vec[j] = st.slider(f"[{best_c}] comparé à [{criteria[j]}] :", 1, 9, 3, key=f"bwm_bo_{j}")

        st.write(f"**À quel point chaque critère est-il plus prioritaire que [{worst_c}] ?**")
        for j in range(num_crit):
            if j != w_idx:
                ow_vec[j] = st.slider(f"[{criteria[j]}] comparé à [{worst_c}] :", 1, 9, 3, key=f"bwm_ow_{j}")

        weights_c, xi = compute_bwm_weights(num_crit, b_idx, w_idx, bo_vec, ow_vec)
        crit_weights = weights_c
        st.info(f"Cohérence BWM : indicateur d'écart ξ* = {xi:.4f}")

# Affichage des poids intermédiaires
df_w = pd.DataFrame({"Critère": criteria, "Poids": crit_weights})
st.dataframe(df_w.set_index("Critère").T.style.format("{:.3f}"), use_container_width=True)

st.markdown("---")

# --- ÉTAPE 3 : QUESTIONNAIRE SUR LES ALTERNATIVES ---
st.header("3. Questionnaire d'évaluation des alternatives")

if eval_strategy.startswith("AHP Hiérarchique"):
    st.write("Pour chaque critère, comparez les alternatives deux à deux.")
    
    # Matrice des priorités locales (lignes: alternatives, colonnes: critères)
    local_priorities = np.zeros((num_alt, num_crit))

    for k, crit_name in enumerate(criteria):
        with st.expander(f"Comparaison des alternatives sous le critère : **{crit_name}**", expanded=(k == 0)):
            mat_alt = np.ones((num_alt, num_alt))
            for i in range(num_alt):
                for j in range(i + 1, num_alt):
                    col_a, col_d = st.columns([2, 2])
                    with col_a:
                        pref_alt = st.radio(
                            f"Concernant **{crit_name}**, quelle option est meilleure ?",
                            [alternatives[i], alternatives[j]],
                            key=f"alt_comp_{k}_{i}_{j}"
                        )
                    with col_d:
                        deg_alt = st.selectbox(
                            f"Degré de préférence :",
                            options=list(saaty_options.keys()),
                            format_func=lambda x: saaty_options[x],
                            key=f"alt_deg_{k}_{i}_{j}"
                        )

                    if pref_alt == alternatives[i]:
                        mat_alt[i, j] = deg_alt
                        mat_alt[j, i] = 1.0 / deg_alt
                    else:
                        mat_alt[i, j] = 1.0 / deg_alt
                        mat_alt[j, i] = deg_alt

            p_loc, _, cr_alt, _ = compute_ahp_vector(mat_alt)
            local_priorities[:, k] = p_loc
            st.caption(f"Ratio de cohérence locale (CR) pour {crit_name} : {cr_alt:.4f}")

    if st.button("Calculer le classement final (Synthèse AHP)", type="primary"):
        # Agrégation AHP : Priorité finale = Somme (poids_critère * priorité_locale)
        final_scores = np.dot(local_priorities, crit_weights)

        df_final = pd.DataFrame({
            "Alternative": alternatives,
            "Score Global": final_scores
        }).sort_values(by="Score Global", ascending=False).reset_index(drop=True)
        df_final["Rang"] = df_final.index + 1

        st.subheader("Classement Final")
        st.dataframe(df_final[["Rang", "Alternative", "Score Global"]].style.format({"Score Global": "{:.4f}"}), use_container_width=True)

        fig = px.bar(df_final, x="Alternative", y="Score Global", color="Alternative", text_auto=".4f", title="Scores synthétiques AHP")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.write("Répondez aux questions qualitatives pour chaque option :")
    
    linguistic_scale = {
        "Très faible / Très mauvais (1)": 1.0,
        "Faible / Mauvais (3)": 3.0,
        "Moyen / Acceptable (5)": 5.0,
        "Bon / Élevé (7)": 7.0,
        "Excellent / Très élevé (9)": 9.0
    }

    # Configuration du sens des critères via questions
    st.write("**Sens d'optimisation des critères :**")
    crit_types = []
    t_cols = st.columns(num_crit)
    for idx, c_col in enumerate(t_cols):
        with c_col:
            val_type = st.selectbox(
                f"Pour **{criteria[idx]}** :",
                ["Bénéfice (Plus c'est élevé, mieux c'est)", "Coût (Plus c'est faible, mieux c'est)"],
                key=f"ctype_q_{idx}"
            )
            crit_types.append(val_type)

    st.write("---")
    generated_matrix = np.zeros((num_alt, num_crit))

    # Questionnaire structuré par alternative
    for i, alt_name in enumerate(alternatives):
        with st.expander(f"Évaluation de : **{alt_name}**", expanded=True):
            cols_eval = st.columns(num_crit)
            for j, crit_name in enumerate(criteria):
                with cols_eval[j]:
                    choice = st.select_slider(
                        f"Niveau en **{crit_name}** :",
                        options=list(linguistic_scale.keys()),
                        value="Moyen / Acceptable (5)",
                        key=f"ling_{i}_{j}"
                    )
                    generated_matrix[i, j] = linguistic_scale[choice]

    rank_engine = st.selectbox("Algorithme d'agrégation :", ["TOPSIS", "WSM (Somme pondérée)"])

    if st.button("Calculer le classement final", type="primary"):
        if rank_engine == "TOPSIS":
            scores = run_topsis_calc(generated_matrix, crit_weights, crit_types)
            lbl = "Score de proximité relative (RC*)"
        else:
            scores = run_wsm_calc(generated_matrix, crit_weights, crit_types)
            lbl = "Score WSM"

        df_res = pd.DataFrame({
            "Alternative": alternatives,
            lbl: scores
        }).sort_values(by=lbl, ascending=False).reset_index(drop=True)
        df_res["Rang"] = df_res.index + 1

        st.subheader("Classement Final")
        st.dataframe(df_res[["Rang", "Alternative", lbl]].style.format({lbl: "{:.4f}"}), use_container_width=True)

        fig2 = px.bar(df_res, x="Alternative", y=lbl, color="Alternative", text_auto=".4f", title=f"Classement obtenu via {rank_engine}")
        st.plotly_chart(fig2, use_container_width=True)
