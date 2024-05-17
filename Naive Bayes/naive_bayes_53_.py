import pandas as pdfrom sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df= pd.read_csv('/content/tourist_data.csv')
df.shape

df['age'].isnull().sum()

scaler = StandardScaler()
df['age'] = scaler.fit_transform(df[['age']])

df

x_df = df.iloc[:, 0:22].values
y_df = df.iloc[:, 22].values

y_df.shape

le = LabelEncoder()

for i in range(x_df.shape[1]):
    x_df[:, i] = le.fit_transform(x_df[:, i])

x_df_treino, x_df_teste, y_df_treino, y_df_teste = train_test_split(x_df, y_df, test_size = 0.20, random_state = 0)

Naive_Bayes = GaussianNB()
Naive_Bayes.fit(x_df_treino,y_df_treino)

predicao = Naive_Bayes.predict(x_df_teste)
predicao

y_df_teste

accuracy_score(y_df_teste, predicao)
