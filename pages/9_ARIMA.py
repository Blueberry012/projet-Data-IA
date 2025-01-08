import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

st.title("Auto-Regressive Integrated Moving Average")

#Importer les données
#df=pd.read_csv('D:\A3\DataScience\data\TP6_dataset.csv')
df=pd.read_csv("data\cleaned_data.csv")
new_columns = ['Pays','Année','Abonnements téléphone (cartes prépayés)','Investissements télécommunications (USD)','Abonnements téléphone (100 habitants)',"Lignes d'accès téléphoniques",'Recettes télécommunication (USD)',"Voies d'accès de communication (100 habitants)"]
df.columns = new_columns
df = df.drop_duplicates()

#df['Pays'].unique()

#Mode
mode = ["Exploration","Guide"]
selected_mode = st.sidebar.selectbox('Choisir un mode :', mode)

if selected_mode =="Exploration": 
    #Exploration Arima
    st.subheader("Paramètres")
    unique_countries = df['Pays'].unique().tolist()
    selected_country = st.selectbox('Choisir un pays :', unique_countries)

    column_names = df.columns.tolist()
    column_names.remove("Pays")
    column_names.remove("Année")
    selected_column = st.selectbox('Choisir une variable :', column_names)
    column_names.remove(selected_column)

    df1=df.copy()
    keepcol=['Pays','Année',selected_column]
    df1 = df1[keepcol]

    #
    df2=df1.copy()
    df2 = df2.drop_duplicates()
    df2 = df2.dropna()
    st.subheader("Dataset")
    st.write(df2)
    
    df3=df2.copy()
    df3 = df3[df3["Pays"] == selected_country]
    del_columns=["Pays"]
    df3 = df3.drop(columns=del_columns)

    df4=df3.copy()
    df4['Année'] = pd.to_datetime(df4['Année'], format='%Y')
    df4.set_index('Année', inplace=True)

    st.subheader("Visualisation")
    plt.figure()
    df4.plot()
    plt.title(selected_column)
    plt.xlabel('Year')
    plt.ylabel(selected_column)
    st.pyplot(plt)

    st.subheader("Test de Stationnarité")
    result = kpss(df4[selected_column])
    st.write('KPSS Statistic:', result[0])
    st.write('p-value:', result[1])

    #model = ARIMA(df4[selected_column], order=(1, 1, 1))
    #model_fit = model.fit()

    #st.write(model_fit.summary())

    df=df3.copy()
    df['Année'] = pd.to_datetime(df['Année'], format='%Y')
    df.set_index('Année', inplace=True)

    # Séparer les données en train (1996-2016) et test (2017-2018)
    train = df[df.index < '2013-01-01']
    test = df[df.index >= '2013-01-01']

    # Entraîner le modèle ARIMA sur l'ensemble d'entraînement
    model = ARIMA(train[selected_column], order=(1, 1, 1))
    model_fit = model.fit()

    # Prédire pour les années 2017 et 2018
    forecast = model_fit.forecast(steps=len(test))
    forecast_index = test.index

    # Évaluer la précision
    st.subheader("Précisiuon du modèle")
    mae = mean_absolute_error(test[selected_column], forecast)
    rmse = np.sqrt(mean_squared_error(test[selected_column], forecast))
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Plot des résultats
    st.subheader("Plot des résultats")
    plt.figure(figsize=(10, 6))
    plt.plot(train.index, train[selected_column], label='Entraînement', color='blue', marker='o')
    plt.plot(test.index, test[selected_column], label='Vraies valeurs (test)', color='green', marker='o')
    plt.plot(forecast_index, forecast, label='Prédictions (test)', color='red', linestyle='--', marker='x')

    plt.title('Prédictions ARIMA vs Vraies Valeurs', fontsize=14)
    plt.xlabel('Année', fontsize=12)
    plt.ylabel(selected_column, fontsize=12)
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

elif selected_mode =="Guide":
    df1=df.copy()
    keepcol=['Pays','Année',"Abonnements téléphone (100 habitants)"]
    df1 = df1[keepcol]

    #
    df2=df1.copy()
    df2 = df2.drop_duplicates()
    df2 = df2.dropna()
    st.subheader("Dataset")
    st.write(df2)

    #
    df3=df2.copy()
    df3 = df3[df3["Pays"] == "France"]
    del_columns=["Pays"]
    df3 = df3.drop(columns=del_columns)

    #
    df4=df3.copy()
    df4['Année'] = pd.to_datetime(df4['Année'], format='%Y')
    df4.set_index('Année', inplace=True)

    #
    st.subheader("Visualisation")
    df4.plot()
    plt.title('Abonnements Téléphone (100 habitants)')
    plt.xlabel('Year')
    plt.ylabel('Abonnements téléphone (100 habitants)')
    plt.show()

    #
    st.subheader("Test de Stationnarité")
    result = kpss(df4['Abonnements téléphone (100 habitants)'])
    st.write('KPSS Statistic:', result[0])
    st.write('p-value:', result[1])

    df=df3.copy()
    df['Année'] = pd.to_datetime(df['Année'], format='%Y')
    df.set_index('Année', inplace=True)

    # Séparer les données en train (1996-2016) et test (2017-2018)
    train = df[df.index < '2013-01-01']
    test = df[df.index >= '2013-01-01']

    # Entraîner le modèle ARIMA sur l'ensemble d'entraînement
    model = ARIMA(train['Abonnements téléphone (100 habitants)'], order=(1, 1, 1))
    model_fit = model.fit()

    # Prédire pour les années 2017 et 2018
    forecast = model_fit.forecast(steps=len(test))
    forecast_index = test.index

    # Évaluer la précision
    st.subheader("Précisiuon du modèle")
    mae = mean_absolute_error(test['Abonnements téléphone (100 habitants)'], forecast)
    rmse = np.sqrt(mean_squared_error(test['Abonnements téléphone (100 habitants)'], forecast))
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Plot des résultats
    st.subheader("Plot des résultats")
    plt.figure(figsize=(10, 6))
    plt.plot(train.index, train['Abonnements téléphone (100 habitants)'], label='Entraînement', color='blue', marker='o')
    plt.plot(test.index, test['Abonnements téléphone (100 habitants)'], label='Vraies valeurs (test)', color='green', marker='o')
    plt.plot(forecast_index, forecast, label='Prédictions (test)', color='red', linestyle='--', marker='x')

    # Titres et légendes
    plt.title('Prédictions ARIMA vs Vraies Valeurs', fontsize=14)
    plt.xlabel('Année', fontsize=12)
    plt.ylabel('Abonnements téléphone (100 habitants)', fontsize=12)
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)