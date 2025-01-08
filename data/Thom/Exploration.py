#CHHUN Thom

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

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
keepcol=['Pays','Année',"Voies d'accès de communication (100 habitants)"]
df1 = df1[keepcol]

#%%
df2=df1.copy()
df2 = df2.drop_duplicates()
df2 = df2.dropna()

#%%
germany_data = df[df["Pays"] == "Allemagne"]
germany_data.set_index("Année", inplace=True)
germany_data.sort_index(inplace=True)
series = germany_data["Voies d'accès de communication (100 habitants)"]

#%%
from statsmodels.tsa.stattools import adfuller

result = adfuller(series.dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])

#%%
diff_series = series.diff().dropna()

#%%
from statsmodels.tsa.arima.model import ARIMA

# Define the ARIMA model
model = ARIMA(series, order=(1, 1, 1))  # Replace (1, 1, 1) with your chosen parameters
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

#%%
forecast = model_fit.forecast(steps=5)  # Forecast the next 5 years
print(forecast)

#%%
import matplotlib.pyplot as plt

# Residual plot
residuals = model_fit.resid
plt.plot(residuals)
plt.title("Residuals")
plt.show()

# Residual histogram
residuals.plot(kind='kde')
plt.title("Residual Distribution")
plt.show()

#%%Exploration KNN
df1=df.copy()
numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
df1 = df.groupby("Pays")[numeric_columns].mean()

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

print(df2.head())
print(df2.info())
print(df2.describe())

df2 = df2.drop_duplicates()
df2 = df2.dropna()
#pas de doublons

#%%
df3=df2.copy()
encoder = LabelEncoder()
df3['Continent'] = encoder.fit_transform(df3['Continent'])

#%%
df4=df3.copy()
del_columns=["Année","Pays"]
df4 = df4.drop(columns=del_columns)

#%%
correlation_matrix = df4.corr()
print("Matrice de corrélation :",correlation_matrix)
sns.heatmap(correlation_matrix, annot=True)
plt.show()

