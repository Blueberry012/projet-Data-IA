#CHHUN Thom

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

#%%
#df=pd.read_csv('D:\A3\DataScience\data\TP6_dataset.csv')
df=pd.read_csv("F:\A3\DataScience\Projet\cleaned_data.csv")

#%%
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

#%%
df5=df4.copy()
num_var=['Abonnements téléphone (cartes prépayés)','Investissements télécommunications (USD)','Abonnements téléphone (100 habitants)',"Lignes d'accès téléphoniques",'Recettes télécommunication (USD)',"Voies d'accès de communication (100 habitants)"]
scaler = MinMaxScaler()
df5[num_var] = scaler.fit_transform(df5[num_var])

#%%
df6=df5.copy()
columns=['Abonnements téléphone (cartes prépayés)','Investissements télécommunications (USD)','Recettes télécommunication (USD)',"Lignes d'accès téléphoniques","Voies d'accès de communication (100 habitants)",'Continent']
X = df6[columns]
y = df6['Abonnements téléphone (100 habitants)']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
model = KNeighborsRegressor(n_neighbors=4)
model.fit(X_train,y_train)

#%%
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)

mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test, predictions)
print("R-squared:", r2)
#Les valeurs faibles de MAE et MSE indiquent une faible erreur entre les prédictions et les valeurs réelles, ce qui montre que votre modèle est précis.
#La valeur de R² de 0.9323 signifie que votre modèle explique 93,23 % de la variabilité de la variable cible, ce qui indique que votre modèle est très bien ajusté aux données.

#%%
# Application de la cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
# cv= 5 ou 10 echantillons
print("Scores :", scores)
print("Score moyen :", scores.mean())

#%%
param_grid = {'n_neighbors': range(1, 21)}
grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='r2')

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score (R-squared):", grid_search.best_score_)

results = pd.DataFrame(grid_search.cv_results_)

plt.plot(results['param_n_neighbors'], results['mean_test_score'], marker='o')

plt.xlabel('Number of Neighbors (n_neighbors)')
plt.ylabel('R-squared Score')
plt.title('R-squared Score vs. Number of Neighbors')
plt.grid(True)
plt.show()

#%%
model = KNeighborsRegressor(n_neighbors=2)
model.fit(X_train,y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)

mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test, predictions)
print("R-squared:", r2)

# Application de la cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
# cv= 5 ou 10 echantillons
print("Scores :", scores)
print("Score moyen :", scores.mean())



