import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D

# Configuration de la page
st.set_page_config(page_title="Régression Linéaire", layout="wide")

st.title("Analyse de Régression Linéaire")
st.write("Cette application permet de réaliser des régressions linéaires simple et multiple sur un jeu de données prédéfini.")

# Charger les données prédéfinies
DATA_PATH = "data\cleaned_data.csv"  # Remplacez par le chemin vers votre fichier CSV
try:
    data = pd.read_csv(DATA_PATH, sep=',')
except FileNotFoundError:
    st.error(f"Le fichier '{DATA_PATH}' est introuvable. Assurez-vous qu'il est présent dans le répertoire du script.")
    st.stop()

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

# Matrice de corrélation
st.subheader("Matrice de Corrélation")
corr_matrix = data.select_dtypes(include=[np.number]).corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Nettoyage des données
data_cleaned = data[[
    "Investissements_telecom", 
    "Abonnements_mobile_100_habitants", 
    "Lignes_acces_telephoniques",
    "Voies_acces_100_habitants"
]].dropna()

# Régression Linéaire Simple
st.subheader("Régression Linéaire Simple")
X_simple = data_cleaned[["Voies_acces_100_habitants"]]
y_simple = data_cleaned["Abonnements_mobile_100_habitants"]

X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)

model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train_simple)

y_pred_simple = model_simple.predict(X_test_simple)
mse_simple = mean_squared_error(y_test_simple, y_pred_simple)
r2_simple = r2_score(y_test_simple, y_pred_simple)

st.write("*MSE (Erreur Quadratique Moyenne):*", mse_simple)
st.write("*R² (Coefficient de Détermination):*", r2_simple)

# Visualisation des résultats
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X_test_simple, y_test_simple, color='blue', label='Données réelles')
ax.plot(X_test_simple, y_pred_simple, color='red', label='Prédictions')
ax.set_title("Régression Linéaire Simple")
ax.set_xlabel("Total des voies d'accès de communication pour 100 habitants")
ax.set_ylabel("Abonnements mobiles pour 100 habitants")
ax.legend()
st.pyplot(fig)

# Régression Linéaire Multiple
st.subheader("Régression Linéaire Multiple")
predictors = ["Investissements_telecom", "Voies_acces_100_habitants"]
target = "Abonnements_mobile_100_habitants"

X = data_cleaned[predictors].values
y = data_cleaned[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write("*MSE (Erreur Quadratique Moyenne):*", mse)
st.write("*R² (Coefficient de Détermination):*", r2)

# Visualisation 3D
st.subheader("Visualisation 3D de la Régression Multiple")
x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)
x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 10)
x1, x2 = np.meshgrid(x1_range, x2_range)
X_grid = np.c_[x1.ravel(), x2.ravel()]
y_pred_grid = model.predict(X_grid).reshape(x1.shape)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Valeurs réelles')
ax.plot_surface(x1, x2, y_pred_grid, color='lightgray', alpha=0.5)
ax.set_title("Régression Linéaire Multiple")
ax.set_xlabel("Investissements télécom")
ax.set_ylabel("Voies d'accès pour 100 habitants")
ax.set_zlabel("Abonnements mobiles pour 100 habitants")
st.pyplot(fig)

# Comparaison Prédictions vs Valeurs Réelles
st.subheader("Comparaison des Prédictions et des Valeurs Réelles")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, color='green', label='Prédictions vs Réelles')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Ligne idéale')
ax.set_title("Prédictions vs Valeurs Réelles")
ax.set_xlabel("Valeurs Réelles")
ax.set_ylabel("Valeurs Prédites")
ax.legend()
st.pyplot(fig)

