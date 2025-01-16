import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("Auto-Regressive Integrated Moving Average")

df=pd.read_csv("data\cleaned_data.csv")
new_columns = ['Pays','Année','Abonnements téléphone (cartes prépayés)','Investissements télécommunications (USD)','Abonnements téléphone (100 habitants)',"Lignes d'accès téléphoniques",'Recettes télécommunication (USD)',"Voies d'accès de communication (100 habitants)","Continent","Développement"]
df.columns = new_columns
df = df.drop_duplicates()

st.header("01 - Préparation des données")
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

df2=df1.copy()
df2 = df2.drop_duplicates()
df2 = df2.dropna()
    
df3=df2.copy()
df3 = df3[df3["Pays"] == selected_country]
del_columns=["Pays"]
df3 = df3.drop(columns=del_columns)

df4=df3.copy()
df4['Année'] = pd.to_datetime(df4['Année'], format='%Y')
df4.set_index('Année', inplace=True)

st.subheader("Dataset")
st.write(df4)

mode = ["Prévision","Précision"]
mode_selected = st.sidebar.selectbox('Choisir le mode :', mode)

if mode_selected =="Prévision":
    st.divider()

    st.header("02 - Visualisation")
    plt.figure()
    df4.plot()
    plt.title('Evolution')
    plt.xlabel('Year')
    st.pyplot(plt)
    st.divider()

    st.header("03 - Test de stationnarité : Augmented Dickey-Fuller (ADF)")
    result = adfuller(df4[selected_column])
    st.write("P-value :", result[1])
    if result[1] <= 0.05:
        st.write("reject the null hypothesis. Data is stationary")
    else:
        st.write("weak evidence against null hypothesis,Data is non-stationary ")

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    plot_acf(df4[selected_column], lags=10, ax=ax[0])
    ax[0].set_title("Autocorrelation Function (ACF)")

    plot_pacf(df4[selected_column], lags=10, ax=ax[1], method="ywm")
    ax[1].set_title("Partial Autocorrelation Function (PACF)")
    plt.tight_layout()
    st.pyplot(fig)
    st.divider()

    st.header("04 - Transformation: Logarithme et Différentiation")
    st.subheader("Visualisation")
    df_trans = df4[selected_column]
    diff = df_trans.diff().dropna()
    df_diff = pd.DataFrame(diff)

    plt.figure()
    df_diff.plot()
    plt.title('Evolution avec différenciation')
    plt.xlabel('Year')
    st.pyplot(plt)

    st.subheader("Test de stationnarité")
    result = adfuller(df_diff[selected_column])
    st.write("P-value :", result[1])
    if result[1] <= 0.05:
        st.write("reject the null hypothesis. Data is stationary")
    else:
        st.write("weak evidence against null hypothesis,Data is non-stationary ")

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    plot_acf(df_diff[selected_column], lags=10, ax=ax[0])
    ax[0].set_title("Autocorrelation Function (ACF)")

    plot_pacf(df_diff[selected_column], lags=10, ax=ax[1], method="ywm")
    ax[1].set_title("Partial Autocorrelation Function (PACF)")
    plt.tight_layout()
    st.pyplot(fig)
    st.divider()

    st.header("05 - Prédiction du modèle")
    # Ajustement du modèle ARIMA
    p,d,q=1,1,1
    model = ARIMA(df_diff, order=(p, d, q))
    model_fit = model.fit()
    #Résumé du modèle
    #st.write(model_fit.summary())

    #Prévisions (par exemple, 5 périodes dans le futur)
    forecast = model_fit.forecast(steps=5)
    #Revenir à l'échelle initiale
    # Inverser la différenciation
    last_value = df_trans.iloc[-1] # Dernière valeur log-transformée avant prédiction
    forecast_original_scale = forecast.cumsum() + last_value
    #forecast_original_scale = np.exp(forecast)

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Prévisions")
        st.write(forecast)
    with col2:
        st.subheader("Prévisions en échelle originale")
        st.write(forecast_original_scale)

    # créer les dates suivantes
    date=pd.date_range(start=df4.index[-1], periods=6, freq='YE')[1:]
    #créer le dataframe de prédictions
    forecast_df = pd.DataFrame({
    "Date": date,
    "Prévisions": forecast_original_scale
    })
    forecast_df.set_index('Date', inplace=True)

    plt.figure(figsize=(10, 6))
    plt.plot(df4.index, df4[selected_column], label=f"{selected_column} réelles")
    plt.plot(forecast_df.index, forecast_df['Prévisions'],
    label="Prévisions", linestyle='--')
    plt.legend()
    plt.title("Prévisions ARIMA")
    plt.xlabel("Date")
    plt.ylabel("Ventes")
    plt.xticks(rotation=45)
    st.pyplot(plt)

elif mode_selected =="Précision":
    train = df4[df4.index < '2018-01-01']
    test = df4[df4.index >= '2018-01-01']

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Train")
        st.write(train)
    with col2:
        st.subheader("Test")
        st.write(test)
    st.divider()

    st.header("02 - Visualisation")
    plt.figure()
    df4.plot()
    plt.title('Evolution')
    plt.xlabel('Year')
    st.pyplot(plt)
    st.divider()

    st.header("03 - Test de stationnarité : Augmented Dickey-Fuller (ADF)")
    result = adfuller(train[selected_column])
    st.write("P-value :", result[1])
    if result[1] <= 0.05:
        st.write("reject the null hypothesis. Data is stationary")
    else:
        st.write("weak evidence against null hypothesis,Data is non-stationary ")

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    plot_acf(train[selected_column], lags=10, ax=ax[0])
    ax[0].set_title("Autocorrelation Function (ACF)")

    plot_pacf(train[selected_column], lags=10, ax=ax[1], method="ywm")
    ax[1].set_title("Partial Autocorrelation Function (PACF)")
    plt.tight_layout()
    st.pyplot(fig)
    st.divider()

    st.header("04 - Transformation: Logarithme et Différentiation")
    st.subheader("Visualisation")
    df_trans = train[selected_column]
    diff = df_trans.diff().dropna()
    df_diff = pd.DataFrame(diff)

    plt.figure()
    df_diff.plot()
    plt.title('Evolution avec différenciation')
    plt.xlabel('Year')
    st.pyplot(plt)

    st.subheader("Test de stationnarité")
    result = adfuller(df_diff[selected_column])
    st.write("P-value :", result[1])
    if result[1] <= 0.05:
        st.write("reject the null hypothesis. Data is stationary")
    else:
        st.write("weak evidence against null hypothesis,Data is non-stationary ")

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    plot_acf(df_diff[selected_column], lags=10, ax=ax[0])
    ax[0].set_title("Autocorrelation Function (ACF)")

    plot_pacf(df_diff[selected_column], lags=10, ax=ax[1], method="ywm")
    ax[1].set_title("Partial Autocorrelation Function (PACF)")
    plt.tight_layout()
    st.pyplot(fig)
    st.divider()

    st.header("05 - Prédiction du modèle")
    # Ajustement du modèle ARIMA
    p,d,q=1,1,1
    model = ARIMA(df_diff['Abonnements téléphone (100 habitants)'], order=(p, d, q))
    model_fit = model.fit()
    #Résumé du modèle
    #print(model_fit.summary())
    forecast = model_fit.forecast(steps=len(test))
    forecast_index = test.index

    #Revenir à l'échelle initiale
    # Inverser la différenciation
    last_value = df_trans.iloc[-1] # Dernière valeur log-transformée avant prédiction
    forecast_original_scale = forecast.cumsum() + last_value
    #forecast_original_scale = np.exp(forecast)


    st.subheader("Prévisions en échelle originale")
    st.write(forecast_original_scale)
    st.subheader("Valeurs Réelles")
    st.write(test)

    # créer les dates suivantes
    date=test.index
    #créer le dataframe de prédictions
    forecast_df = pd.DataFrame({
    "Date": date,
    "Prévisions": forecast_original_scale
    })
    forecast_df.set_index('Date', inplace=True)

    plt.figure(figsize=(10, 6))
    plt.plot(df4.index, df4['Abonnements téléphone (100 habitants)'], label="Abonnements téléphone (100 habitants) réelles", marker='o')
    plt.plot(forecast_df.index, forecast_df['Prévisions'], marker='o',label="Prévisions", linestyle='--', color='red')
    plt.legend()
    plt.title("Prévisions ARIMA")
    plt.xlabel("Date")
    plt.ylabel("Ventes")
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Évaluer la précision
    mae = mean_absolute_error(test['Abonnements téléphone (100 habitants)'], forecast_original_scale)
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
