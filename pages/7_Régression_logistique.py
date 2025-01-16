import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Configuration de la page Streamlit
#st.set_page_config(page_title="Régression Logistique", layout="wide")

st.title("Régression Logistique Interactive")
st.write("Cette application utilise un modèle de régression logistique pour prédire si les abonnements mobiles par 100 habitants sont supérieurs à 100.")

# Chemin vers le fichier nettoyé
DATA_PATH = "data/cleaned_data.csv"

try:
    # Charger les données depuis le fichier CSV
    data = pd.read_csv(DATA_PATH)
    
    # Renommer les colonnes pour simplifier l'analyse
    data.rename(columns={
        "Investissements totaux dans les télécommunications (pour lignes fixes et réseau mobile cellulaire) USD":
            "Investissements_telecom",
        "Total des abonnements au téléphone cellulaire mobile pour 100 habitants":
            "Abonnements_mobile_100_habitants",
        "Total des lignes d'accès téléphoniques":
            "Lignes_acces_telephoniques",
        "Total des voies d'accès de communication pour 100 habitants":
            "Voies_acces_100_habitants"
    }, inplace=True)
    
    # Création de la variable cible
    data['Abonnements_categorie'] = (data['Abonnements_mobile_100_habitants'] > 100).astype(int)
    
    # Nettoyage des données
    data_cleaned = data[[
        "Voies_acces_100_habitants", 
        "Abonnements_categorie"
    ]].dropna()
    
    st.header("Aperçu des données")
    st.write(data_cleaned.head())
    
    # Sélection des variables indépendantes et cible
    X = data_cleaned["Voies_acces_100_habitants"]
    y = data_cleaned["Abonnements_categorie"]
    
    # Mise à l'échelle des données
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.values.reshape(-1, 1))
    
    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Entraînement du modèle
    model_logistic = LogisticRegression()
    model_logistic.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model_logistic.predict(X_test)
    probas = model_logistic.predict_proba(X_test)[:, 1]
    st.divider()
    
    # Affichage des résultats
    st.header("Matrice de Confusion")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    plt.title("Matrice de Confusion")
    plt.xlabel("Prédictions")
    plt.ylabel("Valeurs Réelles")
    st.pyplot(fig)
    
    st.header("Interprétation des données")
    st.text(classification_report(y_test, y_pred))
    st.divider()
    
    # Distribution des probabilités
    st.header("Distribution des Probabilités Prédites")
    fig, ax = plt.subplots()
    ax.hist(probas, bins=20, color='skyblue', edgecolor='black')
    ax.set_title("Distribution des Probabilités Prédites")
    ax.set_xlabel("Probabilité prédite pour la classe 1 (élevé)")
    ax.set_ylabel("Nombre de prédictions")
    st.pyplot(fig)
    st.divider()
    
    # Courbe sigmoïde
    st.header("Courbe Sigmoïde")
    X_range = np.linspace(0, 1, 500).reshape(-1, 1)
    probas_sigmoid = model_logistic.predict_proba(X_range)[:, 1]
    
    fig, ax = plt.subplots()
    ax.plot(X_range, probas_sigmoid, color='red', label='Courbe sigmoïde')
    ax.scatter(X_scaled, y, color='blue', alpha=0.5, label='Données réelles')
    ax.set_title("Courbe Sigmoïde de la Régression Logistique")
    ax.set_xlabel("Voies d'accès de communication (échelle normalisée)")
    ax.set_ylabel("Probabilité prédite de la classe 1")
    ax.legend()
    st.pyplot(fig)
    st.divider()
    
    # Visualisation des données originales
    st.header("Nuage de Points des Données Originales")
    fig, ax = plt.subplots()
    sns.scatterplot(x=scaler.inverse_transform(X_test).flatten(), 
                    y=X_test.flatten(), 
                    hue=y_test, 
                    palette="coolwarm", ax=ax)
    ax.set_title("Nuage de Points")
    ax.set_xlabel("Voies d'accès de communication pour 100 habitants")
    ax.set_ylabel("Valeurs normalisées")
    st.pyplot(fig)

except FileNotFoundError:
    st.error(f"Le fichier {DATA_PATH} est introuvable. Assurez-vous qu'il est placé au bon emplacement.")
except KeyError as e:
    st.error(f"Le fichier ne contient pas la colonne attendue : {e}")
