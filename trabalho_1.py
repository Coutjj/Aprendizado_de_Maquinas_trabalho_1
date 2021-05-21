import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from scipy.stats import pearsonr


conjunto_treino_df = pd.read_csv(
    "eel891-202002-trabalho-1/conjunto_de_treinamento.csv"
)

conjunto_treino_df = conjunto_treino_df.sample(frac=1, random_state=1234)

#print("\nImprimir tipos de dados\n")
#print(conjunto_treino_df.dtypes)

variaveis_categoricas = [
    coluna for coluna in conjunto_treino_df.columns
    if conjunto_treino_df[coluna].dtype == 'object'
]

print("\nVariaveis categoricas\n")
print(variaveis_categoricas)



print('\nVerificar a quantidade de amostras de cada classe:\n')

print(conjunto_treino_df['inadimplente'].value_counts())

print('\nVerificar o valor médio de cada atributo em cada classe:\n')

print(conjunto_treino_df.groupby(['inadimplente']).mean().T)

print("\nVerificar cardinalidade para cada variavel categorica\n")

for var_categorica in variaveis_categoricas:
    print(
        var_categorica + ": "
        + str(len(conjunto_treino_df[var_categorica].unique()))
    )

# forma_envio_solicitacao --> nao ordinal com 3 categorias --> irrelevante
# sexo --> nao ordinal com 4 categorias (M, F, N, Vazio) --> (0, 1, 2)
# estado_onde_nasceu --> nao ordinal com 28 categorias --> (norte, nordeste, centro-oeste, sudeste, sul) (0, 1, 2, 3, 4)
# estado_onde_reside --> nao ordinal com 27 categorias -> (norte, nordeste, centro-oeste, sudeste, sul) (0, 1, 2, 3, 4)
# possui_telefone_residencial --> binaria 2 categorias N, Y --> (0, 1) --> descartado
# codigo_area_telefone_residencial --> nao ordinal 75 categorias --> descartado, estado_onde_reside carrega essa informacao
# possui_telefone_celular --> binaria (N, Y) 1 categoria --> (0, 1) --> descartado
# vinculo_formal_com_empresa --> binaria (N, Y) 2 categorias --> (0, 1)
# estado_onde_trabalha --> nao ordinal 28 categorias --> descartada. todos moram e trabalham no mesmo estado.
# possui_telefone_trabalho --> binaria 2 categorias
# codigo_area_telefone_trabalho --> nao ordinal 67 categorias --> irrelevante

print("\nDescartando variaveis desnecessarias\n")
conjunto_treino_df = conjunto_treino_df.drop([
    'forma_envio_solicitacao',
    'possui_telefone_residencial',
    'codigo_area_telefone_residencial',
    'estado_onde_trabalha',
    'codigo_area_telefone_trabalho',
    'dia_vencimento',
    'possui_telefone_celular'
], axis=1)

# print(conjunto_treino_df.T)

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

conjunto_treino_df = pd.get_dummies(
    conjunto_treino_df,
    columns=['produto_solicitado'],
    prefix='produto_solicitado'
)

#print(conjunto_treino_df)

print('Visualizando dados')

print(
    '\nAplicar binarização simples nas variáveis que tenham'
    'apenas 2 categorias:\n'
)

binarizador = LabelBinarizer()
campo = 'vinculo_formal_com_empresa'
for campo in ['vinculo_formal_com_empresa', 'possui_telefone_trabalho']:
    conjunto_treino_df[campo] = binarizador.fit_transform(conjunto_treino_df[campo])

print('\nVerificar o valor médio de cada atributo em cada classe:\n')

print(conjunto_treino_df.groupby(['inadimplente']).mean().T)

conjunto_treino_df = conjunto_treino_df[conjunto_treino_df['renda_mensal_regular'] < 1000]
conjunto_treino_df = conjunto_treino_df[conjunto_treino_df['renda_extra'] < 1000]
conjunto_treino_df = conjunto_treino_df[conjunto_treino_df['valor_patrimonio_pessoal'] < 30000]

print(conjunto_treino_df)

# cores = [
#     'red' if valor else 'blue'
#     for valor in conjunto_treino_df['inadimplente']
# ]

# grafico = conjunto_treino_df.plot.scatter(
#     'meses_no_trabalho',
#     'renda_mensal_regular',
#     c=cores,
#     s=10,
#     marker='o',
#     alpha=0.5
# )

# plt.show()

conjunto_treino_df.dropna(inplace=True)

atributos_selecionados = [
    'inadimplente',
    # 'id_solicitante',
    # 'produto_solicitado',
    # 'tipo_endereco',
    'idade',
    # 'estado_civil',
    # 'qtde_dependentes',
    # 'grau_instrucao',
    # 'nacionalidade',
    # 'tipo_residencia',
    'meses_na_residencia',
    # 'possui_email',
    'renda_mensal_regular',
    # 'renda_extra',
    # 'possui_cartao_visa',
    # 'possui_cartao_mastercard',
    # 'possui_cartao_diners',
    # 'possui_cartao_amex',
    # 'possui_outros_cartoes',
    # 'possui_telefone_trabalho',
    # 'qtde_contas_bancarias',
    # 'qtde_contas_bancarias_especiais',
    # 'valor_patrimonio_pessoal',
    # 'possui_carro',
    # 'vinculo_formal_com_empresa',
    # 'meses_no_trabalho',
    # 'profissao',
    # 'ocupacao',
    # 'profissao_companheiro',
    # 'grau_instrucao_companheiro',
    # 'local_onde_reside',
    # 'local_onde_trabalha',
    # 'sexo_F',
    # 'sexo_M',
    # 'sexo_N',
    # 'regiao_centro-oeste',
    # 'regiao_nordeste',
    # 'regiao_norte',
    # 'regiao_sudeste',
    # 'regiao_sul',
    # 'produto_solicitado_1',
    # 'produto_solicitado_2',
    # 'produto_solicitado_7',
    # 'grau_instrucao',
    # 'tipo_residencia'
]

conjunto_treino_df = conjunto_treino_df[atributos_selecionados]

dados_treino = conjunto_treino_df.loc[
    :, conjunto_treino_df.columns != 'inadimplente']

dados_alvo = conjunto_treino_df.loc[
    :, conjunto_treino_df.columns == 'inadimplente'].values

dados_alvo = dados_alvo.ravel()

# Ajustando escala

# colunas_escala_padrao = [
#     valor for valor in dados_treino.columns.values.tolist()
#     if valor not in ['renda_mensal_regular', 'idade', 'qtde_dependentes']
# ]

# print(colunas_escala_padrao)

scale = ColumnTransformer(
    transformers=[
        #('mm', MinMaxScaler(), colunas_escala_padrao),
        ('mm2', MinMaxScaler((0,1)), ["renda_mensal_regular"]),
        ('mm3', MinMaxScaler((0, 1)), ['idade']),
    ],
    remainder=MinMaxScaler((0, 1))
)

dados_treino = scale.fit_transform(dados_treino)

print("\nDados corretamente dimendionados\n")
print(dados_treino)


print('\nVerificar o valor médio de cada atributo em cada classe:\n')

print(conjunto_treino_df.groupby(['inadimplente']).mean().T)

print(dados_treino)

cores = [
    'red' if valor else 'blue'
    for valor in conjunto_treino_df['inadimplente']
]

atributos = conjunto_treino_df
atributos = atributos.drop('inadimplente', axis=1)

scatter_matrix = pd.plotting.scatter_matrix(
    atributos,
    c=cores,
    marker='o',
    s=10,
    alpha=0.5,
    diagonal='hist',         # 'hist' ou 'kde'
    hist_kwds={'bins': 20}
)


figura = plt.figure(figsize=(15, 12))

# criar um grafico 3D dentro da figura

grafico = figura.add_subplot(111, projection='3d')

grafico.scatter(
    atributos['renda_mensal_regular'],
    atributos['idade'],
    atributos['meses_na_residencia'],
    c=cores,
    marker='o',
    s=10,
    alpha=1.0
)


# for ax in scatter_matrix.ravel():
#     ax.set_xlabel(ax.get_xlabel(), fontsize=7)
#     ax.set_ylabel(ax.get_ylabel(), fontsize=7)


dados_treino, dados_teste, resposta_treino, resposta_teste = train_test_split(
    dados_treino,
    dados_alvo,
    train_size=0.7
)


classificador = KNeighborsClassifier(
    n_neighbors=13,
    p=2,
    weights='uniform'
)
classificador = classificador.fit(dados_treino, resposta_treino)

conjunto_teste_df = pd.read_csv("./eel891-202002-trabalho-1/conjunto_de_teste.csv")

atributos_selecionados = [
    'idade',
    'meses_na_residencia',
    'renda_mensal_regular',
]

# conjunto_teste_df['meses_na_residencia'] = conjunto_teste_df['meses_na_residencia'].replace(
#     r'^\s*$', '0', regex=True
# )

# conjunto_teste_df['renda_mensal_regular'] = conjunto_teste_df['renda_mensal_regular'].replace(
#     r'^\s*$', '0', regex=True
# )

# conjunto_teste_df['idade'] = conjunto_teste_df['idade'].replace(
#     r'^\s*$', '0', regex=True
# )

conjunto_teste_df.fillna(0, inplace=True)

# conjunto_teste_df.dropna(inplace=True)

id_solicitante = conjunto_teste_df['id_solicitante']
print(id_solicitante)

conjunto_teste_df = conjunto_teste_df[atributos_selecionados]

print(conjunto_teste_df['renda_mensal_regular'].unique())
# print(conjunto_treino_df['sexo'].unique())

conjunto_teste_df = scale.fit_transform(conjunto_teste_df)

# print(conjunto_teste_df)

resposta = classificador.predict(conjunto_teste_df)
resposta = pd.DataFrame(resposta, columns=['inadimplente'])
print(resposta)
resposta = pd.concat([id_solicitante, resposta], axis=1, join='inner')

print("\nRespostas\n")
print(resposta)

resposta.to_csv('./respostas.csv', index=False)


# conjunto_resposta_df = classificador.predict()

# for num_vizinhos in range(1, 50, 2):

#     classificador = KNeighborsClassifier(
#         n_neighbors=num_vizinhos,
#         weights='uniform',
#         p=2
#     )

#     scores = cross_val_score(
#         classificador,
#         dados_treino,
#         dados_alvo,
#         cv=5
#     )

#     print(
#         'nVizinhos = ' + str(num_vizinhos),
#         'scores =', scores,
#         'acurácia média = %6.1f' % (100*sum(scores)/5)
#     )




plt.show()

# conjunto_teste_df = pd.read_csv('eel891-202002-trabalho-1/conjunto_de_teste.csv')

# conjunto_teste_df = conjunto_treino_df.drop([
#     'forma_envio_solicitacao',
#     'possui_telefone_residencial',
#     'codigo_area_telefone_residencial',
#     'estado_onde_trabalha',
#     'possui_telefone_trabalho',
#     'codigo_area_telefone_trabalho',
#     'dia_vencimento',
#     'possui_telefone_celular'
# ], axis=1)

# conjunto_treino_df['sexo'] = conjunto_treino_df['sexo'].replace(
#     r'^\s*$', 'N', regex=True
# )

# conjunto_teste_df = pd.get_dummies(
#     conjunto_treino_df,
#     columns=['sexo'],
#     prefix='sexo'
# )

# conjunto_teste_df['regiao'] = conjunto_teste_df['estado_onde_reside'].map(dict_estados_regioes)

# conjunto_teste_df = conjunto_teste_df.drop([
#     'estado_onde_nasceu',
#     'estado_onde_reside',
# ], axis=1)

# conjunto_teste_df = pd.get_dummies(
#     conjunto_teste_df,
#     columns=['regiao'],
#     prefix='regiao'
# )

# binarizador = LabelBinarizer()
# campo = 'vinculo_formal_com_empresa'
# conjunto_teste_df[campo] = binarizador.fit_transform(conjunto_teste_df[campo])


# conjunto_resposta.df = classificador.predict(conjunto_teste_df)