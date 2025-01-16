import streamlit as st
print(st.__version__)

CURRENT_THEME = "blue"
IS_DARK_THEME = True

st.title('Projet Data Science')

st.header("Optimisation de l'allocation de la bande passante par apprentissage automatique")

st.image("image/telecom.jpg", width=500, use_container_width=False, clamp=True)

st.subheader("Description du problème:")

st.write("Les fournisseurs de services de télécommunications sont constamment confrontés à la nécessité d'optimiser la bande passante pour répondre efficacement à la demande fluctuante des utilisateurs. La gestion inefficace de la bande passante peut entraîner une dégradation de la qualité de service, des retards ou même des interruptions de service, ce qui affecte la satisfaction client et peut entraîner des pertes financières significatives.")

st.subheader("Contexte spécifique:")

st.write("Avec l'augmentation exponentielle des données mobiles et de l'utilisation d'internet reportée par les statistiques de l'OCDE, ainsi que le développement de nouvelles applications gourmandes en données comme le streaming vidéo, les jeux en ligne, et la réalité augmentée, il devient crucial de développer des solutions plus dynamiques et intelligentes pour la gestion de la bande passante.")

st.subheader("Objectif Général")

st.write("Ce projet vise à permettre aux étudiants de mettre en pratique les concepts enseignés en cours en appliquant des méthodes d’analyse et de modélisation des données. Les étudiants travailleront en groupe de 3 et utiliseront les jeux de données définis dans le sites : ")

url = "https://stats.oecd.org/Index.aspx?DataSetCode=TEL"
#st.write("check out this [link](%s)" % url)
st.markdown("check out this [link](%s)" % url)

st.write("L’objectif est d’explorer, nettoyer, analyser et développer un modèle de machine learning capable de prédire la demande de bande passante en temps réel et d'ajuster automatiquement la répartition des ressources réseau pour maximiser l'efficacité et la qualité de service en suivant les étapes décrites ci-dessous.")

st.divider()

st.write("N'hésitez pas à parcourir notre application pour en découvrir sur la Data Science")
