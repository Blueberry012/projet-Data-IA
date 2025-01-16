import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate

#st.set_page_config(layout="wide")
st.title("K-Nearest Neaighbors")

#df=pd.read_csv('D:\A3\DataScience\data\TP6_dataset.csv')
df=pd.read_csv("data/cleaned_data.csv")
new_columns = ['Pays','Année','Abonnements téléphone (cartes prépayés)','Investissements télécommunications (USD)','Abonnements téléphone (100 habitants)',"Lignes d'accès téléphoniques",'Recettes télécommunication (USD)',"Voies d'accès de communication (100 habitants)","Continent","Développement"]
df.columns = new_columns
df = df.drop_duplicates()

df3=df.copy()
encoder = LabelEncoder()
df3['Continent'] = encoder.fit_transform(df3['Continent'])
df3['Développement'] = encoder.fit_transform(df3['Développement'])
#df3['Pays'] = encoder.fit_transform(df3['Pays'])

df4=df3.copy()
del_columns=["Année"]
df4 = df4.drop(columns=del_columns)
df4 = df4.dropna()

df5=df4.copy()
num_var=['Abonnements téléphone (cartes prépayés)','Investissements télécommunications (USD)','Abonnements téléphone (100 habitants)',"Lignes d'accès téléphoniques",'Recettes télécommunication (USD)',"Voies d'accès de communication (100 habitants)"]
scaler = MinMaxScaler()
df5[num_var] = scaler.fit_transform(df5[num_var])

mode = ["KNN Classification","KNN Régression"]
mode_selected = st.sidebar.selectbox('Choisir le mode :', mode)

if mode_selected =="KNN Classification":
    st.header("01 - Préparation des données")
    st.subheader("Dataset")
    st.write(df5)

    df6=df5.copy()
    columns=['Développement','Continent','Abonnements téléphone (cartes prépayés)','Investissements télécommunications (USD)','Recettes télécommunication (USD)',"Lignes d'accès téléphoniques","Voies d'accès de communication (100 habitants)",'Abonnements téléphone (100 habitants)']
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
    st.divider()

    st.header("02 - Nombre de k voisins optimal")
    param_grid = {'n_neighbors': range(1, 21)}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    st.subheader("GridSearch")
    results = pd.DataFrame(grid_search.cv_results_)

    plt.plot(results['param_n_neighbors'], results['mean_test_score'], marker='o')
    plt.xlabel('Nombre de voisins (n_neighbors)')
    plt.ylabel('Score de précision (Accuracy)')
    plt.title('Précision vs Nombre de voisins')
    plt.grid(True)
    st.pyplot(plt)

    #st.write("Meilleurs paramètres:", grid_search.best_params_)
    st.write("Meilleur score (Accuracy):", grid_search.best_score_)
    st.divider()

    st.header("03 - Exploration")
    k= st.slider('k-plus proches voisins', 1, 20)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train,y_train)
    accuracy = model.score(X_test, y_test)
    st.write(f"Précision du modèle : {accuracy:.2f}")
    y_pred = model.predict(X_test)

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Prédictions")
        st.write(y_pred)
    with col2:
        st.subheader("Valeurs réelles")
        st.write(y_test)
    st.divider()

    st.header("04 - Validation croisée")
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    # cv= 5 ou 10 echantillons
    #st.write("Scores :", scores)
    st.write("Précision moyenne :", scores.mean())

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Prédictions')
    plt.ylabel('Réel')
    plt.title('Matrice de Confusion')
    st.pyplot(plt)


elif mode_selected =="KNN Régression":
    st.header("01 - Préparation des données")
    st.subheader("Dataset")
    #df5['Pays'] = encoder.fit_transform(df5['Pays'])
    #del_columns=["Pays","Développement","Continent"]
    del_columns=["Pays"]
    df5 = df5.drop(columns=del_columns)
    st.write(df5)

    column_names = df5.columns.tolist()
    #column_names.remove("Pays")
    column_names.remove("Continent")
    column_names.remove("Développement")
    selected_column = st.selectbox('Choisir une variable à prédire:', column_names)

    column_names = df5.columns.tolist()
    column_names.remove(selected_column)

    df6=df5.copy()
    #columns=['Abonnements téléphone (cartes prépayés)','Investissements télécommunications (USD)','Recettes télécommunication (USD)',"Lignes d'accès téléphoniques","Voies d'accès de communication (100 habitants)",'Continent']
    X = df6[column_names]
    y = df6[selected_column]
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("X_train")
        st.write(X_train)
    with col2:
        st.subheader("y_train")
        st.write(y_train)
    st.divider()

    st.header("02 - Nombre de k voisins optimal")
    st.subheader("GridSearch")
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

    #st.write("Best parameters:", grid_search.best_params_)
    st.write("Best score (R-squared):", grid_search.best_score_)
    st.divider()

    st.header("03 - Exploration")
    k= st.slider('k-plus proches voisins', 1, 20)
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    st.write("R-squared :", r2)
    mae = mean_absolute_error(y_test, predictions)
    st.write("Mean Absolute Error :", mae)
    mse = mean_squared_error(y_test, predictions)
    st.write("Mean Squared Error :", mse)
    
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Prédictions") 
        st.write(predictions)
    with col2:
        st.subheader("Valeurs réelles")
        st.write(y_test)
    st.divider()

    st.header("04 - Validation croisée")
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    # cv= 5 ou 10 echantillons
    #st.write("R-squared :", scores)
    st.write("R-squared moyen :", scores.mean())
    
    scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error']
    results = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)
    mae_scores = -results['test_neg_mean_absolute_error']
    mse_scores = -results['test_neg_mean_squared_error']
    st.write("Mean Absolute Error moyen :", mae_scores.mean())
    st.write("Mean Squared Error moyen :", mse_scores.mean())
