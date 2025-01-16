import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np


st.title("K-Means")


# Import des données
df=pd.read_csv("data/cleaned_data.csv")

#Mode
mode = ["Exploration","Guide"]
selected_mode = st.sidebar.selectbox('Choisir un mode :', mode)

isExplorationMode=True

if selected_mode == "Exploration":
    isExplorationMode=True
else:
    isExplorationMode=False
    a=2000
    k=3
    

# On choisit l'année que l'on souhaite analyser



st.header("01 - Nombre de clusters")

if isExplorationMode==True:
    a= st.slider('Choisir une année', 1996, 2018)
df_per_year = df[(df["TIME_PERIOD"]==a)]
df_per_year=df_per_year.drop('TIME_PERIOD',axis=1)



# On va remplacer les valeurs NaN par une proportion de la valeur manquante

dfkmean=df_per_year[["Pays","Total des voies d'accès de communication pour 100 habitants","Total des abonnements au téléphone cellulaire mobile pour 100 habitants"]]

dfkmean["Total des abonnements au téléphone cellulaire mobile pour 100 habitants"] = dfkmean[
    "Total des abonnements au téléphone cellulaire mobile pour 100 habitants"
].fillna(
    dfkmean["Total des voies d'accès de communication pour 100 habitants"] / 7.4
)

dfkmean["Total des voies d'accès de communication pour 100 habitants"] = dfkmean[
    "Total des voies d'accès de communication pour 100 habitants"
].fillna(
    dfkmean["Total des abonnements au téléphone cellulaire mobile pour 100 habitants"] * 7.4
)

# Si les deux sont NaN, on supprime la ligne
dfkmean = dfkmean.dropna(subset=[
    "Total des abonnements au téléphone cellulaire mobile pour 100 habitants",
    "Total des voies d'accès de communication pour 100 habitants"
], how='all')



# Méthode elbow afin de déterminer le nombre de clusters idéal

st.subheader("Méthode elbow afin de déterminer le nombre de clusters idéal")

W = []
for i in range(1, 15):
    km = KMeans(n_clusters = i, init = 'k-means++')
    km.fit(dfkmean[["Total des voies d'accès de communication pour 100 habitants","Total des abonnements au téléphone cellulaire mobile pour 100 habitants"]])
    W.append(km.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 15), W)
plt.grid(True)
plt.xlabel('Nombre k de clusters')
if isExplorationMode==False:
    plt.axvline(x=3, color='red', linestyle='--')
plt.ylabel('Somme des distances au carré')

st.pyplot(plt)

st.divider()

st.header("02  —  Résultats")

# On choisit le nombre de cluster que l'on souhaite

if isExplorationMode==True:
    k= st.slider('Choisir le nombre de cluster(s) que vous souhaitez', 1, len(dfkmean.index))

st.subheader("Visualisation")

# On fait le clustering K-Means

# Ajoute une colonne Cluster pour savoir a quel cluster le pays appartient

km = KMeans(n_clusters = k, init="k-means++")
dfkmean['Cluster']= km.fit_predict(dfkmean[["Total des voies d'accès de communication pour 100 habitants","Total des abonnements au téléphone cellulaire mobile pour 100 habitants"]])

# Tracer les points avec une couleur par cluster
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    dfkmean["Total des voies d'accès de communication pour 100 habitants"], 
    dfkmean["Total des abonnements au téléphone cellulaire mobile pour 100 habitants"], 
    c=dfkmean['Cluster'], cmap='viridis', marker='o'
)

# Tracer les centres des clusters
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 
            s=200, c='red', marker='X', label="Centres des clusters")

# Ajouter une légende pour les clusters
for cluster_id in range(k):  # k étant le nombre de clusters
    plt.scatter([], [], color=scatter.cmap(scatter.norm(cluster_id)), label=f'Cluster {cluster_id}')

plt.legend()
plt.title("K-Means - Vue des clusters")
plt.xlabel("Total des voies d'accès de communication pour 100 habitants")
plt.ylabel("Total des abonnements au téléphone cellulaire mobile pour 100 habitants")
st.pyplot(plt)



# Affichage des pays et de leur clusteur respectif
st.subheader("Dataset")

st.write(dfkmean)
