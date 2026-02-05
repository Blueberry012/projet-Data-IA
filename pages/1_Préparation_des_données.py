import streamlit as st
import pandas as pd

st.title("Nettoyage et Préparation des données")

st.header("01 - Exportation des données")
url = "https://stats.oecd.org/Index.aspx?DataSetCode=TEL"
st.markdown("Veuillez trouver les données [ici](%s)" % url)
df=pd.read_csv("data/data.csv")
st.write(df)


st.header("02 - Transposition des données")
columns=['Pays','Série','TIME_PERIOD','OBS_VALUE']
df2=df.copy()
df2 = df[columns]
df3=df2.copy()
df3= df.pivot(index=["Pays", "TIME_PERIOD"], columns="Série", values="OBS_VALUE").reset_index()
st.write(df3)


st.header("03 - Informations Générales")
df4=df3.copy()
#df4 = df4[(df4["TIME_PERIOD"] >= 2007) & (df4["TIME_PERIOD"] <= 2017)]
#st.write(df4.info())
st.write(df4.describe())


st.header("04 - Nombres de données manquantes par variables")
nan_count=df4.isna().sum()
st.write(nan_count)
df5=df4.copy()
del_columns=["Nombre d’abonnés à la télévision par câble","Total Internet Protocol (IP) telephone subscriptions"]
df5 = df5.drop(columns=del_columns)

df6=df5.copy()
df6 = df6[df6["Pays"] != "OCDE - Total"]


st.header("05 - Enrichissement des données")
df7=df6.copy()
new_columns = ['Pays','Année','Abonnements téléphone (cartes prépayés)','Investissements télécommunications (USD)','Abonnements téléphone (100 habitants)',"Lignes d'accès téléphoniques",'Recettes télécommunication (USD)',"Voies d'accès de communication (100 habitants)"]
df7.columns = new_columns

pays = [
    'Allemagne', 'Australie', 'Autriche', 'Belgique', 'Canada', 'Chili', 
    'Colombie', 'Corée', 'Danemark', 'Espagne', 'Estonie', 'Finlande', 
    'France', 'Grèce', 'Hongrie', 'Irlande', 'Islande', 'Israël', 
    'Italie', 'Japon', 'Lettonie', 'Lituanie', 'Luxembourg', 'Mexique', 
    'Norvège', 'Nouvelle-Zélande', 'Pays-Bas', 'Pologne', 'Portugal', 
    'Royaume-Uni', 'République slovaque', 'Slovénie', 'Suisse', 'Suède', 
    'Tchéquie', 'Türkiye', 'États-Unis'
]

continents = [
    'Europe', 'Océanie', 'Europe', 'Europe', 'Amérique du Nord', 'Amérique du Sud', 
    'Amérique du Sud', 'Asie', 'Europe', 'Europe', 'Europe', 'Europe', 
    'Europe', 'Europe', 'Europe', 'Europe', 'Europe', 'Asie', 
    'Europe', 'Asie', 'Europe', 'Europe', 'Europe', 'Amérique du Nord', 
    'Europe', 'Océanie', 'Europe', 'Europe', 'Europe', 
    'Europe', 'Europe', 'Europe', 'Europe', 'Europe', 
    'Europe', 'Asie', 'Amérique du Nord'
]

developpement = [
    "pays développé",  # Allemagne
    "pays développé",  # Australie
    "pays développé",  # Autriche
    "pays développé",  # Belgique
    "pays développé",  # Canada
    "pays en développement",  # Chili
    "pays en développement",  # Colombie
    "pays développé",  # Corée
    "pays développé",  # Danemark
    "pays développé",  # Espagne
    "pays développé",  # Estonie
    "pays développé",  # Finlande
    "pays développé",  # France
    "pays développé",  # Grèce
    "pays développé",  # Hongrie
    "pays développé",  # Irlande
    "pays développé",  # Islande
    "pays développé",  # Israël
    "pays développé",  # Italie
    "pays développé",  # Japon
    "pays développé",  # Lettonie
    "pays développé",  # Lituanie
    "pays développé",  # Luxembourg
    "pays en développement",  # Mexique
    "pays développé",  # Norvège
    "pays développé",  # Nouvelle-Zélande
    "pays développé",  # Pays-Bas
    "pays développé",  # Pologne
    "pays développé",  # Portugal
    "pays développé",  # Royaume-Uni
    "pays développé",  # République slovaque
    "pays développé",  # Slovénie
    "pays développé",  # Suisse
    "pays développé",  # Suède
    "pays développé",  # Tchéquie
    "pays en développement",  # Türkiye
    "pays développé"  # États-Unis
]

df_continent = pd.DataFrame({'Pays': pays, 'Continent': continents, 'Développement': developpement})
df7 = pd.merge(df7, df_continent, on='Pays')
st.write(df7)

