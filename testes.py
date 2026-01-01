import pandas as pd

df = pd.read_excel('primeirasemanasetembro.xlsx')
""" df_frequencia_milhar = df[df['milhar'].notnull()]
df_frequencia_milhar = df_frequencia_milhar[['milhar']].reset_index(drop=True).value_counts().head(10)
print(df_frequencia_milhar) """

df_grupo = df[df['dezena'].notnull()]
print(df_grupo)
df_grupo = df_grupo[['dezena']].value_counts().head(10)
print(df_grupo)
""" df_frequencia_grupo = df[df['dezena']].reset_index(drop=True).mode()
print(df_frequencia_grupo) """