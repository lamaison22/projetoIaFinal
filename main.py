
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier       
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
data = pd.read_csv('all_games.csv')
data.drop(data.columns[3], axis=1, inplace=True)
print(data)


# Transformando os nomes dos jogos
label_encoder_game = LabelEncoder()
data['name'] = label_encoder_game.fit_transform(data['name'])

# Transformando as plataformas
label_encoder_platform = LabelEncoder()
data['platform'] = label_encoder_platform.fit_transform(data['platform'])

# Transformando as datas
data['release_date'] = pd.to_datetime(data['release_date'])
data['release_year'] = data['release_date'].dt.year
data['release_month'] = data['release_date'].dt.month
data['release_day'] = data['release_date'].dt.day

# Excluindo colunas originais se necessário
# data = data.drop(columns=['game_name', 'platform', 'release_date'])

# Visualizando as primeiras linhas do dataframe transformado
print(data.head())

##########################transformando de volta em nomes#########################################
name = data['name'][0]
original_game_name = label_encoder_game.inverse_transform([name])[0]
print(f"ID do jogo: {name}, Nome original do jogo: {original_game_name}")

# Codificando uma plataforma de volta ao seu valor original
platform = data['platform'][0]
original_platform = label_encoder_platform.inverse_transform([platform])[0]
print(f"ID da plataforma: {platform}, Plataforma original: {original_platform}")
###############################################################################################


# Separando os dados em variáveis de entrada (X) e saída (y)
# X = data[['release_year', 'release_month', 'release_day']]
# y = data['platform']

# # Dividindo os dados em conjunto de treinamento e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Criando e treinando um modelo de rede neural
# mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
# mlp.fit(X_train, y_train)

# # Avaliando o modelo
# accuracy = mlp.score(X_test, y_test)
# print(f"Acurácia do modelo: {accuracy:.2f}")


X = data[['release_year', 'release_month', 'release_day']]
y = data['platform']

# Dividindo os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=60)

# Criando e treinando um modelo de Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Fazendo previsões
y_pred = rf.predict(X_test)

# Avaliando o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")