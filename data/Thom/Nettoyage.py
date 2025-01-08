#CHHUN Thom

import pandas as pd

#1)
#df=pd.read_csv('D:\A3\DataScience\data\TP6_dataset.csv')
df=pd.read_csv("D:\A3\DataScience\Projet\data.csv")

#%% Partie
columns=['Pays','Série','TIME_PERIOD','OBS_VALUE']
df2=df.copy()
df2 = df[columns]
print(df2)

df3=df2.copy()
df3= df.pivot(index=["Pays", "TIME_PERIOD"], columns="Série", values="OBS_VALUE").reset_index()

#%% Partie
df4=df3.copy()
#df4 = df4[(df4["TIME_PERIOD"] >= 2007) & (df4["TIME_PERIOD"] <= 2017)]

#%% Partie
print(df4.info())
print(df4.describe())

nan_count=df4.isna().sum()
#%% Partie
df5=df4.copy()
del_columns=["Nombre d’abonnés à la télévision par câble","Total Internet Protocol (IP) telephone subscriptions"]
df5 = df5.drop(columns=del_columns)

#%% Partie
df6=df5.copy()
df6 = df6[df6["Pays"] != "OCDE - Total"]

#%% Partie
df7=df6.copy()

df7.to_csv('D:\A3\DataScience\Projet\cleaned_data.csv', index=False)

