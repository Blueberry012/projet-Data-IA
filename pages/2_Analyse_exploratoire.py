import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder

def Monde(dataframe):
    center = [46.603354, 2.668561]
    zoom = 2
    pays_data = dataframe
    rad = 10
    return center, zoom, pays_data, rad


def Pays(dataframe, pays):
    pays_data = dataframe[dataframe['Pays'] == pays]
    center = [pays_data['Latitude'].mean(), pays_data['Longitude'].mean()]
    zoom = 6
    rad = 50
    return center, zoom, pays_data, rad


def circle_marker(map, row, rad, icon_color, popup):
    folium.CircleMarker(location=[row['Latitude'], row['Longitude']],
                        stroke=False,
                        radius=rad,
                        color=icon_color,
                        fill=True,
                        fill_color=icon_color,
                        fill_opacity=0.7,
                        popup=popup,
                        icon=folium.Icon(color=icon_color, icon='')).add_to(map)

st.title("Analyse Exploratoire")

df=pd.read_csv("data\cleaned_data2.csv")

mode = ["Carte","Statistique Descriptive"]
mode_selected = st.sidebar.selectbox('Choisir le mode :', mode)

if mode_selected =="Carte":
    df_map=df.copy()

    pays = [
        'Allemagne', 'Australie', 'Autriche', 'Belgique', 'Canada', 'Chili', 
        'Colombie', 'Corée', 'Danemark', 'Espagne', 'Estonie', 'Finlande', 
        'France', 'Grèce', 'Hongrie', 'Irlande', 'Islande', 'Israël', 
        'Italie', 'Japon', 'Lettonie', 'Lituanie', 'Luxembourg', 'Mexique', 
        'Norvège', 'Nouvelle-Zélande', 'Pays-Bas', 'Pologne', 'Portugal', 
        'Royaume-Uni', 'République slovaque', 'Slovénie', 'Suisse', 'Suède', 
        'Tchéquie', 'Türkiye', 'États-Unis'
    ]

    longitude = [
        10.4515, 133.7751, 13.1998, 4.4699, -106.3468, -71.5429, 
        -74.2973, 127.7669, 9.5018, -3.7038, 25.0136, 25.7482, 
        2.2137, 21.8243, 19.5033, -8.2439, -19.0208, 34.8516, 
        12.5674, 138.2529, 24.6032, 23.8813, 6.1296, -102.5528, 
        8.4689, 174.8859, 5.2913, 19.1451, -8.2245, -3.4359, 
        19.6990, 14.9955, 8.2275, 18.6435, 15.4720, 35.2433, -95.7129
    ]

    latitude = [
        51.1657, -25.2744, 47.5162, 50.5039, 56.1304, -35.6751, 
        4.5709, 35.9078, 56.2639, 40.4168, 58.5953, 61.9241, 
        46.6034, 39.0742, 47.1625, 53.4129, 64.9631, 31.0461, 
        41.8719, 36.2048, 56.8796, 55.1694, 49.8153, 23.6345, 
        60.4720, -40.9006, 52.1326, 51.9194, 39.3999, 55.3781, 
        48.6690, 46.1512, 46.8182, 60.1282, 49.8175, 38.9637, 37.0902
    ]

    map = pd.DataFrame({'Pays': pays, 'Longitude': longitude, 'Latitude': latitude})
    df_map = pd.merge(df_map, map, on='Pays')

    column_names = df.columns.tolist()
    column_names.remove("Pays")
    column_names.remove("Continent")
    column_names.remove("Développement")
    column_names.remove("Année")
    selected_column = st.selectbox('Choisir une variable à étudier :', column_names)

    df_map = df_map.dropna(subset=[selected_column])
    columns_keep = ['Pays','Année',selected_column,'Continent','Développement','Longitude','Latitude']
    df_map = df_map[columns_keep]

    annee= st.slider('Année', 1996, 2018)
    df_map = df_map[(df_map["Année"] == annee)]

    pays = df_map['Pays'].unique().tolist()
    pays = sorted(pays)
    pays.insert(0, "Monde")

    df_map['Latitude'] = df_map['Latitude'].astype(float)
    df_map['Longitude'] = df_map['Longitude'].astype(float)

    df_map['Catégorie'], bins =  pd.qcut(df_map[selected_column], 3, labels=False, retbins=True)
    quantile_ranges = []
    for start, end in zip(bins[:-1], bins[1:]):
        start, end = int(start), int(end)
        formatted_range = f"{start} à {end}"
        quantile_ranges.append(formatted_range)

    df_map["Catégorie"] = df_map["Catégorie"].astype(str)
    purpose_colour = {
            '0': '#55E2E9',
            '1': '#0496C7',
            '2': '#02367B'
        }
    categ = st.sidebar.radio('Répartition',('Tous',quantile_ranges[0],quantile_ranges[1],quantile_ranges[2]))
    st.write(df_map)

    country = st.selectbox("Choisissez une ville", pays, key='ville')
    if country=="Monde":
        center, zoom, pays_data, rad = Monde(df_map)
    else:
        center, zoom, pays_data, rad = Pays(df_map, country)
    map = folium.Map(location=center, zoom_start=zoom, control_scale=True)

    data2=df_map
    if categ == quantile_ranges[0]:
        data2=data2[(data2['Catégorie']== '0')]
    elif categ == quantile_ranges[1]:
        data2=data2[(data2['Catégorie']== '1')]
    elif categ == quantile_ranges[2]:
        data2=data2[(data2['Catégorie']== '2')]
    else:
        data2=df_map

    for i,row in data2.iterrows():
        content = f'Pays : {str(row["Pays"])}<br>' f'{selected_column} : {str(row[selected_column])}'
        iframe = folium.IFrame(content, width=400, height=55)
        popup = folium.Popup(iframe, min_width=400, max_width=500)
                            
        try:
            icon_color = purpose_colour[row['Catégorie']]
        except:
            icon_color = 'gray'
        circle_marker(map, row, rad, icon_color, popup)
    st_folium(map, height=500, width=700, key='map')

    df_graph=df.copy()
    del_columns=["Continent", "Développement"]
    df_graph = df_graph.drop(columns=del_columns)
    if country=="Monde":
        del_columns=["Pays"]
        df_graph = df_graph.drop(columns=del_columns)
        df_graph = df_graph.groupby(['Année']).mean()
    else:
        df_graph = df_graph[(df_graph["Pays"] == country)]

    df_graph['Année'] = df_graph.index
    st.write(df_graph)

    plt.plot(df_graph['Année'], df_graph[selected_column], marker='o', label=selected_column, color='blue')
    plt.title(f"{selected_column} au cours du temps")
    plt.xlabel('Année')
    plt.grid(True)
    st.pyplot(plt)


elif mode_selected =="Statistique Descriptive":
    column_names = df.columns.tolist()
    column_names.remove("Pays")
    column_names.remove("Continent")
    column_names.remove("Développement")
    column_names.remove("Année")
    selected_column = st.selectbox('Choisir une variable à étudier :', column_names)

    st.header("01 - Distribution")
    plt.figure(figsize=(8, 5))
    sns.histplot(df[selected_column], kde=True, color="blue")
    plt.title(f"Distribution {selected_column}")
    plt.xlabel(selected_column)
    plt.ylabel("Fréquence")
    st.pyplot(plt)
    st.divider()

    st.header("02 - Relation")
    column_names = df.columns.tolist()
    column_names.remove("Pays")
    column_names.remove("Continent")
    column_names.remove("Développement")
    column_names.remove("Année")
    column_names.remove(selected_column)
    selected_column2 = st.selectbox('Choisir une variable à étudier:', column_names)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=selected_column, y=selected_column2, hue="Continent")
    plt.title(f"Relation entre {selected_column} et {selected_column2}")
    plt.xlabel(selected_column)
    plt.ylabel(selected_column2)
    st.pyplot(plt)
    st.divider()

    st.header("03 - Normalisation des données")
    scaler = StandardScaler()
    columns_to_scale = [
        "Abonnements téléphone (cartes prépayés)",
        "Investissements télécommunications (USD)",
        "Abonnements téléphone (100 habitants)",
        "Lignes d'accès téléphoniques",
        "Recettes télécommunication (USD)",
        "Voies d'accès de communication (100 habitants)"
    ]
    df_scaled = df.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    st.write("Données après centrage et mise à l'échelle :")
    st.write(df_scaled.head())

    # Transformation logarithmique si nécessaire (exemple pour une variable)
    df["Log"] = np.log1p(df[selected_column])

    plt.figure(figsize=(8, 5))
    sns.histplot(df["Log"], kde=True, color="green")
    plt.title(f"Distribution {selected_column} (échelle logarithmique)")
    plt.xlabel("Log({selected_column})")
    plt.ylabel("Fréquence")
    st.pyplot(plt)

    st.divider()
    st.header("04 - Matrice de corrélation")
    df3=df.copy()
    encoder = LabelEncoder()
    df3['Continent'] = encoder.fit_transform(df3['Continent'])
    df3['Développement'] = encoder.fit_transform(df3['Développement'])

    df4=df3.copy()
    del_columns=["Pays"]
    df4 = df4.drop(columns=del_columns)

    correlation_matrix = df4.corr()
    #print("Matrice de corrélation :",correlation_matrix)
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True)
    st.pyplot(plt)
