import pandas as pd
import numpy as np
import datetime as dt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_excel('C:\\Users\\Ivina\\Desktop\\jogodobicho\\jogodobicho\\primeirasemanasetembro.xlsx')

# organizar por data
df = df.sort_values(by='data')

# trabalhar apenas com dezena
df_milhar = df[['milhar']].copy()
df_horario = df[['horario']].copy()
df_datas = df[['data']].copy()

print(f"Total de registros: {len(df_milhar)}")

# normalização
normalizador = MinMaxScaler(feature_range=(0, 1))
df_milhar_normalizado = normalizador.fit_transform(df_milhar)

# criar janelas deslizantes
tamanho_janela = 10  # Últimos 10 resultados para prever o próximo
previsao, valor_real = [], []

for i in range(tamanho_janela, len(df_milhar_normalizado)):
    seq = df_milhar_normalizado[i - tamanho_janela:i, 0]  # Janela de 10 valores
    previsao.append(seq)
    valor_real.append(df_milhar_normalizado[i, 0])  # Próximo valor

# Converte para arrays numpy
previsao = np.array(previsao)
valor_real = np.array(valor_real)

# Reshape para LSTM [samples, timesteps, features]
previsao = previsao.reshape((previsao.shape[0], previsao.shape[1], 1))

print(f"Formato das sequências: {previsao.shape}")
print(f"Formato dos valores reais: {valor_real.shape}")

# Dividir em treino e teste (80/20)
tam_treinamento = int(len(previsao) * 0.8)
x_treinamento = previsao[:tam_treinamento]
y_treinamento = valor_real[:tam_treinamento]

x_teste = previsao[tam_treinamento:]
y_teste = valor_real[tam_treinamento:]

print(f"\nDados de treinamento: {len(x_treinamento)}")
print(f"Dados de teste: {len(x_teste)}")

# Criar modelo LSTM
modelo = Sequential()
modelo.add(LSTM(units=50, return_sequences=True, input_shape=(tamanho_janela, 1)))
modelo.add(Dropout(0.2))
modelo.add(LSTM(units=50))
modelo.add(Dropout(0.2))
modelo.add(Dense(units=1))

modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
modelo.summary()
modelo.save('jogobicho_milhar.keras')
# Treinar modelo
historico = modelo.fit(x_treinamento, y_treinamento, 
                       batch_size=16, 
                       epochs=200, 
                       verbose=1,
                       validation_data=(x_teste, y_teste))

# Avaliar no conjunto de teste
previsao_teste = modelo.predict(x_teste)
previsao_teste_desnormalizada = normalizador.inverse_transform(previsao_teste)
y_teste_desnormalizado = normalizador.inverse_transform(y_teste.reshape(-1, 1))

mae = mean_absolute_error(y_teste_desnormalizado, previsao_teste_desnormalizada)
rmse = np.sqrt(mean_squared_error(y_teste_desnormalizado, previsao_teste_desnormalizada))

print(f"\n=== AVALIAÇÃO DO MODELO ===")
print(f"Erro Médio Absoluto (MAE): {mae:.2f}")
print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f}")

# Comparação teste
print("\n=== COMPARAÇÃO (Teste) - Primeiros 10 ===")
comparacao = pd.DataFrame({
    'milhar_real': y_teste_desnormalizado[:10, 0].astype(int),
    'milhar_prevista': previsao_teste_desnormalizada[:10, 0].astype(int),
    'diferenca': np.abs(y_teste_desnormalizado[:10, 0] - previsao_teste_desnormalizada[:10, 0]).astype(int)
})
print(comparacao)

# Fazer previsões futuras
n = 10  # Próximos 10 resultados
janela_atual = df_milhar_normalizado[-tamanho_janela:].reshape(1, tamanho_janela, 1)
previsoes_futuras = []

for _ in range(n):
    proxima_pred = modelo.predict(janela_atual, verbose=0)[0, 0]
    previsoes_futuras.append(proxima_pred)
    # Atualiza janela: remove primeiro, adiciona novo
    janela_atual = np.append(janela_atual[:, 1:, :], [[[proxima_pred]]], axis=1)

previsoes_futuras = np.array(previsoes_futuras).reshape(-1, 1)
previsoes_futuras_desnormalizadas = normalizador.inverse_transform(previsoes_futuras)

# Gerar datas futuras
ultima_data = pd.to_datetime(df_datas.iloc[-1, 0])
datas_futuras = [ultima_data + datetime.timedelta(days=i+1) for i in range(n)]

# DataFrame com previsões
df_previsoes = pd.DataFrame({
    'data': datas_futuras,
    'milhar_prevista': previsoes_futuras_desnormalizadas[:, 0].astype(int)
})

print("\n=== PREVISÕES FUTURAS ===")
print(df_previsoes)

# Análise de frequência (bônus)
print("\n=== TOP 10 milhar MAIS FREQUENTES (Histórico) ===")
frequencia = df['milhar'].value_counts().head(10)
print(frequencia)