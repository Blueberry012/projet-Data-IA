#CHHUN Thom

import pandas as pd
import matplotlib.pyplot as plt

#%%
#df=pd.read_csv('D:\A3\DataScience\data\TP6_dataset.csv')
df=pd.read_csv("F:\A3\DataScience\Projet\cleaned_data.csv")

#%%
new_columns = ['Pays','Année','Abonnements téléphone (cartes prépayés)','Investissements télécommunications (USD)','Abonnements téléphone (100 habitants)',"Lignes d'accès téléphoniques",'Recettes télécommunication (USD)',"Voies d'accès de communication (100 habitants)"]
df.columns = new_columns
df = df.drop_duplicates()

#%%
df['Pays'].unique()

#%%Exploration Arima
df1=df.copy()
keepcol=['Pays','Année',"Abonnements téléphone (100 habitants)"]
df1 = df1[keepcol]

#%%
df2=df1.copy()
df2 = df2.drop_duplicates()
df2 = df2.dropna()

#%%
df3=df2.copy()
df3 = df3[df3["Pays"] == "France"]
del_columns=["Pays"]
df3 = df3.drop(columns=del_columns)

#%%$
df4=df3.copy()
df4['Année'] = pd.to_datetime(df4['Année'], format='%Y')
df4.set_index('Année', inplace=True)

#%%$
df4.plot()
plt.title('Abonnements Téléphone (100 habitants)')
plt.xlabel('Year')
plt.ylabel('Abonnements téléphone (100 habitants)')
plt.show()

#%%$
from statsmodels.tsa.stattools import adfuller
result = adfuller(df4['Abonnements téléphone (100 habitants)'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

#%%$
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df4['Abonnements téléphone (100 habitants)'], order=(1, 1, 1))
model_fit = model.fit()

print(model_fit.summary())

#%%$
forecast = model_fit.forecast(steps=5)  # Forecasting next 5 years
print(forecast)

#%%$
plt.plot(df4.index, df4['Abonnements téléphone (100 habitants)'], label='Actual')
plt.plot(pd.date_range(df4.index[-1], periods=6, freq='Y')[1:], forecast, label='Forecast', color='red')
plt.title('ARIMA Forecast for Abonnements Téléphone')
plt.xlabel('Year')
plt.ylabel('Abonnements téléphone (100 habitants)')
plt.legend()
plt.show()

#%%$
forecast_steps = 5
forecast = model_fit.forecast(steps=forecast_steps)
forecast_index = pd.date_range(df.index[-1], periods=forecast_steps + 1, freq='Y')[1:]

plt.plot(df4.index, df4['Abonnements téléphone (100 habitants)'], label='Actual Data', color='blue', marker='o')

# Plot the forecasted data
plt.plot(forecast_index, forecast, label='Forecast', color='red', marker='x')

# Add shaded area for the confidence interval (95% by default)
conf_int = model_fit.get_forecast(steps=forecast_steps).conf_int()
plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='gray', alpha=0.2)

# Title and labels
plt.title('ARIMA Forecast for Abonnements Téléphone', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Abonnements téléphone (100 habitants)', fontsize=12)
plt.legend()

#%%$
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

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
mae = mean_absolute_error(test['Abonnements téléphone (100 habitants)'], forecast)
rmse = np.sqrt(mean_squared_error(test['Abonnements téléphone (100 habitants)'], forecast))
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Plot des résultats
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
plt.show()