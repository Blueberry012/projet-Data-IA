import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

#st.set_page_config(layout="wide")

st.title("K-Nearest Neaighbors")

#Importer les données
#df=pd.read_csv('D:\A3\DataScience\data\TP6_dataset.csv')
df=pd.read_csv("data\cleaned_data.csv")
new_columns = ['Pays','Année','Abonnements téléphone (cartes prépayés)','Investissements télécommunications (USD)','Abonnements téléphone (100 habitants)',"Lignes d'accès téléphoniques",'Recettes télécommunication (USD)',"Voies d'accès de communication (100 habitants)"]
df.columns = new_columns
df = df.drop_duplicates()

#%%
df1=df.copy()
#numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
#df1 = df.groupby("Pays")[numeric_columns].mean()

#%%
df2=df1.copy()
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

df_continent = pd.DataFrame({'Pays': pays, 'Continent': continents})
df2 = pd.merge(df2, df_continent, on='Pays')

#%%
df3=df2.copy()
encoder = LabelEncoder()
df3['Continent'] = encoder.fit_transform(df3['Continent'])

#%%
df4=df3.copy()
del_columns=["Année"]
df4 = df4.drop(columns=del_columns)
df4 = df4.dropna()

df5=df4.copy()
num_var=['Abonnements téléphone (cartes prépayés)','Investissements télécommunications (USD)','Abonnements téléphone (100 habitants)',"Lignes d'accès téléphoniques",'Recettes télécommunication (USD)',"Voies d'accès de communication (100 habitants)"]
scaler = MinMaxScaler()
df5[num_var] = scaler.fit_transform(df5[num_var])

#Mode
knn = ["KNN Classification","KNN Régression"]
selected_knn = st.sidebar.selectbox('Choisir le KNN :', knn)

#Mode
mode = ["Exploration","Guide"]
selected_mode = st.sidebar.selectbox('Choisir un mode :', mode)

if selected_mode =="Exploration":
    st.header(selected_knn)
    st.subheader("Dataset")
    st.write(df2)

    #Heatmap
    st.subheader("Heatmap")
    df_test=df4.copy()
    del_columns=["Pays"]
    df_test = df_test.drop(columns=del_columns)
    correlation_matrix = df_test.corr()
    #print("Matrice de corrélation :",correlation_matrix)

    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, ax=ax)
    st.pyplot(fig)

    if selected_knn =="KNN Régression":
        column_names = df5.columns.tolist()
        column_names.remove("Pays")
        selected_column = st.selectbox('Choisir une variable :', column_names)
        column_names.remove(selected_column)

        df6=df5.copy()
        #columns=['Abonnements téléphone (cartes prépayés)','Investissements télécommunications (USD)','Recettes télécommunication (USD)',"Lignes d'accès téléphoniques","Voies d'accès de communication (100 habitants)",'Continent']
        X = df6[column_names]
        y = df6[selected_column]
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #
        k= st.slider('k-plus proches voisins', 1, 20)
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train,y_train)

        #
        predictions = model.predict(X_test)

        col1,col2 = st.columns(2)
        with col1:
            st.write("Prédictions :", predictions)
        with col2:
            st.write("Valeurs réelles :", y_test)

        mae = mean_absolute_error(y_test, predictions)
        st.write("Mean Absolute Error :", mae)
        mse = mean_squared_error(y_test, predictions)
        st.write("Mean Squared Error :", mse)
        r2 = r2_score(y_test, predictions)
        st.write("R-squared :", r2)
        #Les valeurs faibles de MAE et MSE indiquent une faible erreur entre les prédictions et les valeurs réelles, ce qui montre que votre modèle est précis.
        #La valeur de R² de 0.9323 signifie que votre modèle explique 93,23 % de la variabilité de la variable cible, ce qui indique que votre modèle est très bien ajusté aux données.
        
        # Application de la cross-validation
        st.subheader("Application de la cross-validation")
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        # cv= 5 ou 10 echantillons
        st.write("R-squared :", scores)
        st.write("R-squared moyen :", scores.mean())
    
    if selected_knn =="KNN Classification":
        df6=df5.copy()
        columns=['Abonnements téléphone (cartes prépayés)','Investissements télécommunications (USD)','Recettes télécommunication (USD)',"Lignes d'accès téléphoniques","Voies d'accès de communication (100 habitants)",'Continent','Abonnements téléphone (100 habitants)']
        X = df6[columns]
        y = df6['Pays']
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        k= st.slider('k-plus proches voisins', 1, 20)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train,y_train)

        accuracy = model.score(X_test, y_test)
        st.write(f"Précision du modèle : {accuracy:.2f}")

        y_pred = model.predict(X_test)

        col1,col2 = st.columns(2)
        with col1:
            st.write("Prédictions :", y_pred)
        with col2:
            st.write("Valeurs réelles :", y_test)

        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)

        # Création du graphique
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.xlabel('Prédictions')
        plt.ylabel('Réel')
        plt.title('Matrice de Confusion')
        st.pyplot(plt)

elif selected_mode =="Guide":
    st.header(selected_knn)

    if selected_knn =="KNN Régression":
        df6=df5.copy()
        columns=['Abonnements téléphone (cartes prépayés)','Investissements télécommunications (USD)','Recettes télécommunication (USD)',"Lignes d'accès téléphoniques","Voies d'accès de communication (100 habitants)",'Continent']
        X = df6[columns]
        y = df6['Abonnements téléphone (100 habitants)']
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        col1,col2 = st.columns(2)
        with col1:
            st.subheader("X_train")
            st.write(X_train)
        with col2:
            st.subheader("y_train")
            st.write(y_train)

        st.subheader("GridSearch pour optimiser le k-plus proches voisins")

        param_grid = {'n_neighbors': range(1, 21)}
        grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='r2')

        grid_search.fit(X_train, y_train)

        results = pd.DataFrame(grid_search.cv_results_)

        fig, ax = plt.subplots()
        ax.plot(results['param_n_neighbors'], results['mean_test_score'], marker='o')
        ax.set_xlabel('Number of Neighbors (n_neighbors)')
        ax.set_ylabel('R-squared Score')
        ax.set_title('R-squared Score vs. Number of Neighbors')
        ax.grid(True)
        st.pyplot(fig)

        st.write("Best parameters:", grid_search.best_params_)
        st.write("Best score (R-squared):", grid_search.best_score_)

        # Application de la cross-validation
        st.subheader("Application de la cross-validation")
        model = KNeighborsRegressor(n_neighbors=2)
        model.fit(X_train,y_train)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        # cv= 5 ou 10 echantillons
        st.write("R-squared :", scores)
        st.write("R-squared moyen :", scores.mean())

    elif selected_knn =="KNN Classification":
        df6=df5.copy()
        columns=['Abonnements téléphone (cartes prépayés)','Investissements télécommunications (USD)','Recettes télécommunication (USD)',"Lignes d'accès téléphoniques","Voies d'accès de communication (100 habitants)",'Continent','Abonnements téléphone (100 habitants)']
        X = df6[columns]
        y = df6['Pays']
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        col1,col2 = st.columns(2)
        with col1:
            st.subheader("X_train")
            st.write(X_train)
        with col2:
            st.subheader("y_train")
            st.write(y_train)

        param_grid = {'n_neighbors': range(1, 21)}

        # Effectuer la recherche par validation croisée
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        st.subheader("GridSearch pour optimiser le k-plus proches voisins")
        # Résultats sous forme de DataFrame
        results = pd.DataFrame(grid_search.cv_results_)

        # Graphique des scores
        plt.plot(results['param_n_neighbors'], results['mean_test_score'], marker='o')

        plt.xlabel('Nombre de voisins (n_neighbors)')
        plt.ylabel('Score de précision (Accuracy)')
        plt.title('Précision vs Nombre de voisins')
        plt.grid(True)
        st.pyplot(plt)

        st.subheader("Application de la cross-validation")
        # Meilleurs paramètres et score
        st.write("Meilleurs paramètres:", grid_search.best_params_)
        st.write("Meilleur score (Accuracy):", grid_search.best_score_)