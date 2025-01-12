import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import numpy as np

st.title("HCA")

# Import des données
df=pd.read_csv("data\cleaned_data.csv")

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
    selected_linkage_mode="Ward"
    selected_metric_mode="Euclidean"


# On choisit l'année que l'on souhaite analyser



if isExplorationMode==True:
    a= st.slider('Choisir une année', 1996, 2018)
df_per_year = df[(df["TIME_PERIOD"]==a)]
df_per_year=df_per_year.drop('TIME_PERIOD',axis=1)


if isExplorationMode==True:
    # Heatmap afin de voir quelles variables sont intéréssantes à étudier entre-elles
    st.subheader("HeatMap pour les variables \"Total des voies d'accès de communication pour 100 habitants\" ET \"Total des abonnements au téléphone cellulaire mobile pour 100 habitants\" selon l'année.")
    correlation_matrix = df_per_year[["Total des voies d'accès de communication pour 100 habitants","Total des abonnements au téléphone cellulaire mobile pour 100 habitants"]].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        
    plt.title("Matrice de corrélation")
    st.pyplot(plt)



# On va remplacer les valeurs NaN par une proportion de la valeur manquante

dfhca=df_per_year[["Pays","Total des voies d'accès de communication pour 100 habitants","Total des abonnements au téléphone cellulaire mobile pour 100 habitants"]]

dfhca["Total des abonnements au téléphone cellulaire mobile pour 100 habitants"] = dfhca[
    "Total des abonnements au téléphone cellulaire mobile pour 100 habitants"
].fillna(
    dfhca["Total des voies d'accès de communication pour 100 habitants"] / 7.4
)

dfhca["Total des voies d'accès de communication pour 100 habitants"] = dfhca[
    "Total des voies d'accès de communication pour 100 habitants"
].fillna(
    dfhca["Total des abonnements au téléphone cellulaire mobile pour 100 habitants"] * 7.4
)

# Si les deux sont NaN, on supprime la ligne
dfhca = dfhca.dropna(subset=[
    "Total des abonnements au téléphone cellulaire mobile pour 100 habitants",
    "Total des voies d'accès de communication pour 100 habitants"
], how='all')



# Choisir les modes de distances à appliquer 

if isExplorationMode==True:
    linkage_mode = ["Complete","Average","Ward","Single"]
    selected_linkage_mode = st.selectbox('Choisir la distance linkage :', linkage_mode)

    metric_mode = ["Euclidean", "L1", "L2", "Manhattan", "Cosine"]
    selected_metric_mode = st.selectbox('Choisir la distance metric :', metric_mode, disabled=(selected_linkage_mode == "Ward"))

    if(selected_linkage_mode=="Ward"):
        selected_metric_mode="Euclidean"

# On fait le clustering HCA

linkage_matrix = linkage(dfhca[["Total des voies d'accès de communication pour 100 habitants","Total des abonnements au téléphone cellulaire mobile pour 100 habitants"]], method=selected_linkage_mode.lower())



# Dendogramme
plt.figure(figsize=(8, 6))
dendrogram(linkage_matrix, labels=dfhca["Pays"].to_list())

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
st.pyplot(plt)



# On choisit le nombre de cluster que l'on souhaite
if isExplorationMode==True:
    k= st.slider('Choisir le nombre de cluster(s) que vous souhaitez', 1, len(dfhca.index))

HCA = AgglomerativeClustering(n_clusters=k, metric=selected_metric_mode.lower(), linkage=selected_linkage_mode.lower())
Labels = HCA.fit_predict(dfhca[["Total des voies d'accès de communication pour 100 habitants", 
                                "Total des abonnements au téléphone cellulaire mobile pour 100 habitants"]])

# Création du graphique
plt.figure(figsize=(8, 6))

# Tracer les points avec une couleur par cluster
scatter = plt.scatter(
    dfhca["Total des voies d'accès de communication pour 100 habitants"], 
    dfhca["Total des abonnements au téléphone cellulaire mobile pour 100 habitants"], 
    c=Labels, cmap='viridis', marker='o'
)

# Ajouter une légende pour les clusters
for cluster_id in range(k):  # k étant le nombre de clusters
    plt.scatter([], [], color=scatter.cmap(scatter.norm(cluster_id)), label=f'Cluster {cluster_id}')

# Ajouter des labels et titre
plt.title("Clustering hiérarchique - Vue des clusters")
plt.xlabel("Total des voies d'accès de communication pour 100 habitants")
plt.ylabel("Total des abonnements au téléphone cellulaire mobile pour 100 habitants")
plt.legend()

# Afficher le graphique
st.pyplot(plt)

dfhca['Cluster'] = Labels

st.write(dfhca)
