import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from core.ahp import compute_ahp_weights
from core.bwm import compute_bwm_weights
from core.ranking import run_wsm, run_topsis

st.set_page_config(page_title="Système MCDM - Aide à la Décision", layout="wide")

st.title("Système d'Aide à la Décision Multicritère (MCDM)")
st.markdown("Ce portail permet d'évaluer les poids subjectifs des critères puis de classer vos alternatives.")

# Initialisation du session_state
if "weights" not in st.session_state:
    st.session_state.weights = None
if "criteria" not in st.session_state:
    st.session_state.criteria = ["Coût", "Qualité", "Délai"]

# --- ÉTAPE 1 : CONFIGURATION DES CRITÈRES ---
st.header("1. Définition des critères")
criteria_input = st.text_input(
    "Noms des critères (séparés par une virgule) :",
    value=", ".join(st.session_state.criteria)
)
criteria = [c.strip() for c in criteria_input.split(",") if c.strip()]
num_criteria = len(criteria)

if num_criteria < 2:
    st.warning("Veuillez renseigner au moins 2 critères pour continuer.")
    st.stop()

# --- ÉTAPE 2 : QUESTIONNAIRE DE PONDÉRATION SUBJECTIVE ---
st.header("2. Pondération subjective des critères")
method = st.radio("Sélectionnez la méthode de pondération :", ["AHP (Saaty)", "BWM (Best-Worst Method)"])

scale_labels = {
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

if method == "AHP (Saaty)":
    st.subheader("Questionnaire de comparaison par paires (AHP)")
    st.write("Comparez l'importance relative de chaque paire de critères.")

    pairwise_matrix = np.ones((num_criteria, num_criteria))

    for i in range(num_criteria):
        for j in range(i + 1, num_criteria):
            col1, col2 = st.columns([2, 1])
            with col1:
                preference = st.radio(
                    f"Entre **{criteria[i]}** et **{criteria[j]}**, lequel est le plus important ?",
                    [criteria[i], criteria[j]],
                    key=f"pref_{i}_{j}"
                )
            with col2:
                intensity = st.selectbox(
                    f"Degré d'importance :",
                    options=list(scale_labels.keys()),
                    format_func=lambda x: scale_labels[x],
                    key=f"val_{i}_{j}"
                )

            if preference == criteria[i]:
                pairwise_matrix[i, j] = intensity
                pairwise_matrix[j, i] = 1.0 / intensity
            else:
                pairwise_matrix[i, j] = 1.0 / intensity
                pairwise_matrix[j, i] = intensity

    if st.button("Calculer les poids AHP"):
        weights, ci, cr, is_consistent = compute_ahp_weights(pairwise_matrix)
        st.session_state.weights = weights

        st.write("### Résultats de cohérence")
        c1, c2, c3 = st.columns(3)
        c1.metric("Indice de Cohérence (CI)", f"{ci:.4f}")
        c2.metric("Ratio de Cohérence (CR)", f"{cr:.4f}")
        c3.metric("État", "Cohérent" if is_consistent else "Incohérent", delta="OK" if is_consistent else "-Non respecté")

        if not is_consistent:
            st.error("Le ratio de cohérence CR est supérieur à 10% (0.10). Il est recommandé de revoir certaines comparaisons.")

elif method == "BWM (Best-Worst Method)":
    st.subheader("Configuration Best-Worst (BWM)")

    c_best, c_worst = st.columns(2)
    with c_best:
        best_criterion = st.selectbox("Sélectionnez le meilleur critère (Best) :", criteria, index=0)
    with c_worst:
        worst_criterion = st.selectbox("Sélectionnez le pire critère (Worst) :", criteria, index=num_criteria - 1)

    if best_criterion == worst_criterion:
        st.error("Le meilleur critère et le pire critère doivent être distincts.")
        st.stop()

    best_idx = criteria.index(best_criterion)
    worst_idx = criteria.index(worst_criterion)

    st.write("---")
    st.write(f"#### Vecteur Best-to-Others (BO) : Importance de **{best_criterion}** par rapport aux autres")
    bo_vector = np.ones(num_criteria)
    for j in range(num_criteria):
        if j == best_idx:
            bo_vector[j] = 1.0
        else:
            val = st.slider(
                f"À quel point **{best_criterion}** est-il plus important que **{criteria[j]}** ?",
                min_value=1,
                max_value=9,
                value=3,
                key=f"bo_{j}"
            )
            bo_vector[j] = val

    st.write("---")
    st.write(f"#### Vecteur Others-to-Worst (OW) : Importance des critères par rapport à **{worst_criterion}**")
    ow_vector = np.ones(num_criteria)
    for j in range(num_criteria):
        if j == worst_idx:
            ow_vector[j] = 1.0
        else:
            val = st.slider(
                f"À quel point **{criteria[j]}** est-il plus important que **{worst_criterion}** ?",
                min_value=1,
                max_value=9,
                value=3,
                key=f"ow_{j}"
            )
            ow_vector[j] = val

    if st.button("Calculer les poids BWM"):
        try:
            weights, xi_star = compute_bwm_weights(num_criteria, best_idx, worst_idx, bo_vector, ow_vector)
            st.session_state.weights = weights
            st.metric("Indicateur d'incohérence optimal (ξ*)", f"{xi_star:.4f}")
            if xi_star > 0.25:
                st.warning("Valeur de ξ* élevée : vérifiez la cohérence entre vos deux vecteurs BO et OW.")
        except Exception as e:
            st.error(f"Erreur d'optimisation : {e}")

# Affichage des poids calculés
if st.session_state.weights is not None:
    df_weights = pd.DataFrame({
        "Critère": criteria,
        "Poids": st.session_state.weights
    })
    
    st.write("### Poids calculés pour chaque critère")
    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        fig = px.bar(df_weights, x="Critère", y="Poids", title="Distribution des poids des critères", text_auto=".3f")
        st.plotly_chart(fig, use_container_width=True)
    with col_table:
        st.dataframe(df_weights.style.format({"Poids": "{:.4f}"}), use_container_width=True)

    # --- ÉTAPE 3 : CLASSEMENT DES ALTERNATIVES (« LE RESTE ») ---
    st.write("---")
    st.header("3. Évaluation et classement des alternatives")

    st.write("Précisez la typologie de chaque critère :")
    crit_types = []
    type_cols = st.columns(num_criteria)
    for idx, col in enumerate(type_cols):
        with col:
            t = st.selectbox(criteria[idx], ["Bénéfice (+)", "Coût (-)"], key=f"ctype_{idx}")
            crit_types.append(t)

    alternatives_input = st.text_input("Noms des alternatives (séparés par une virgule) :", value="A1, A2, A3")
    alternatives = [a.strip() for a in alternatives_input.split(",") if a.strip()]

    st.write("Saisissez les performances de chaque alternative :")
    default_data = np.ones((len(alternatives), num_criteria)) * 10.0
    df_matrix = pd.DataFrame(default_data, index=alternatives, columns=criteria)
    
    edited_matrix = st.data_editor(df_matrix, use_container_width=True)

    ranking_method = st.selectbox("Méthode de classement :", ["WSM (Somme pondérée)", "TOPSIS"])

    if st.button("Calculer le classement final"):
        dec_matrix = edited_matrix.to_numpy()
        w = st.session_state.weights

        if ranking_method == "WSM (Somme pondérée)":
            scores = run_wsm(dec_matrix, w, crit_types)
            metric_col = "Score WSM"
        else:
            scores = run_topsis(dec_matrix, w, crit_types)
            metric_col = "Proximité relative (RC*)"

        df_rank = pd.DataFrame({
            "Alternative": alternatives,
            metric_col: scores
        }).sort_values(by=metric_col, ascending=False).reset_index(drop=True)
        
        df_rank["Rang"] = df_rank.index + 1

        st.subheader("Classement final")
        st.dataframe(df_rank[["Rang", "Alternative", metric_col]].style.format({metric_col: "{:.4f}"}), use_container_width=True)