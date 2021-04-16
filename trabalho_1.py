import pandas as pd
import numpy as nd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


conjunto_treino_df = pd.read_csv("eel891-202002-trabalho-1/conjunto_de_treinamento.csv")

print("Imprime conjunto de dados de treino antes do embaralhamento\n")
print(conjunto_treino_df)

conjunto_treino_df = conjunto_treino_df.sample(frac=1, random_state=1234)
print("Imprime dados apos embaralhamento\n")
print(conjunto_treino_df)

# print("Tabela de treinamento transposta")
# print(conjunto_treino_df.T)

print("\nImprimir tipos de dados\n")
print(conjunto_treino_df.dtypes)

variaveis_categoricas = [
    linha for linha in conjunto_treino_df.columns
    if conjunto_treino_df[linha].dtype == 'object'
]

print("\nVariaveis categoricas\n")
print(variaveis_categoricas)

print("\nVerificar cardinalidade para cada variavel categorica\n")

for var_categorica in variaveis_categoricas:
    print(
        var_categorica + ": "
        + str(len(conjunto_treino_df[var_categorica].unique()))
    )

print(len(conjunto_treino_df[conjunto_treino_df['vinculo_formal_com_empresa'] == 'Y'].index))
# forma_envio_solicitacao --> nao ordinal com 3 categorias --> irrelevante
# sexo --> nao ordinal com 4 categorias (M, F, N, Vazio) --> (0, 1, 2)
# estado_onde_nasceu --> nao ordinal com 28 categorias --> (norte, nordeste, centro-oeste, sudeste, sul) (0, 1, 2, 3, 4)
# estado_onde_reside --> nao ordinal com 27 categorias -> (norte, nordeste, centro-oeste, sudeste, sul) (0, 1, 2, 3, 4)
# possui_telefone_residencial --> binaria 2 categorias N, Y --> (0, 1)
# codigo_area_telefone_residencial --> nao ordinal 75 categorias --> descartado, estado_onde_reside carrega essa informacao
# possui_telefone_celular --> binaria (N, Y) 1 categoria --> (0, 1)
# vinculo_formal_com_empresa --> binaria (N, Y) 2 categorias --> (0, 1)
# estado_onde_trabalha --> nao ordinal 28 categorias --> descartada. todos moram e trabalham no mesmo estado.
# possui_telefone_trabalho --> binaria 2 categorias --> irrelevante
# codigo_area_telefone_trabalho --> nao ordinal 67 categorias --> irrelevante

print("\nDescartando variaveis desnecessarias\n")
conjunto_treino_df = conjunto_treino_df.drop([
    'forma_envio_solicitacao',
    'possui_telefone_residencial',
    'codigo_area_telefone_residencial',
    'estado_onde_trabalha',
    'possui_telefone_trabalho',
    'codigo_area_telefone_trabalho',
    'dia_vencimento',
    'possui_telefone_celular'
], axis=1)

print(conjunto_treino_df.T)

print("\nTrocando campo de Vazio para N na coluina sexo\n")

conjunto_treino_df['sexo'] = conjunto_treino_df['sexo'].replace(
    r'^\s*$', 'N', regex=True
)

print(conjunto_treino_df['sexo'].unique())

print('\nAplicando One-hot encoding na coluna sexo\n')

conjunto_treino_df = pd.get_dummies(
    conjunto_treino_df,
    columns=['sexo'],
    prefix='sexo'
)

print(conjunto_treino_df[list(conjunto_treino_df.filter(regex='sexo*'))])

print("\nMapeando estados\n")

dict_regioes = {
    'norte': ['AM', 'PA', 'AC', 'RO', 'RR', 'AP', 'TO'],
    'nordeste': ['MA', 'PI', 'CE', 'RN', 'PB', 'PE', 'AL', 'SE', 'BA'],
    'centro-oeste': ['GO', 'MT', 'MS', 'DF'],
    'sul': ['PR', 'SC', 'RS'],
    'sudeste': ['SP', 'RJ', 'MG', 'ES']
}

# inversao chave valor
dict_estados_regioes = dict()
for regiao in dict_regioes:
    for estado in dict_regioes[regiao]:
        dict_estados_regioes[estado] = regiao

conjunto_treino_df['regiao'] = conjunto_treino_df['estado_onde_reside'].map(dict_estados_regioes)

print(conjunto_treino_df)


print('\nRemovendo colunas contendo informacao dos estados\n')

conjunto_treino_df = conjunto_treino_df.drop([
    'estado_onde_nasceu',
    'estado_onde_reside',
], axis=1)

print(conjunto_treino_df)

print('\nOne-hot encoding regiao\n')

conjunto_treino_df = pd.get_dummies(
    conjunto_treino_df,
    columns=['regiao'],
    prefix='regiao'
)

print(conjunto_treino_df)

