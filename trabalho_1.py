import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score


conjunto_treino_df = pd.read_csv(
    "eel891-202002-trabalho-1/conjunto_de_treinamento.csv"
)

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

print(len(conjunto_treino_df[conjunto_treino_df['vinculo_formal_com_empresa'] == 'Y'].index))
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

#print(conjunto_treino_df.T)

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

conjunto_treino_df = conjunto_treino_df[conjunto_treino_df['renda_mensal_regular'] < 5000]
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
    #'id_solicitante',
    #'produto_solicitado',
    #'tipo_endereco',
    'idade',
    'estado_civil',
    #'qtde_dependentes',
    #'grau_instrucao',
    #'nacionalidade',
    #'tipo_residencia',
    #'meses_na_residencia',
    #'possui_email',
    'renda_mensal_regular',
    #'renda_extra',
    #'possui_cartao_visa',
    #'possui_cartao_mastercard',
    #'possui_cartao_diners',
    #'possui_cartao_amex',
    #'possui_outros_cartoes',
    #'possui_telefone_trabalho',
    #'qtde_contas_bancarias',
    #'qtde_contas_bancarias_especiais',
    #'valor_patrimonio_pessoal',
    #'possui_carro',
    #'vinculo_formal_com_empresa',
    #'meses_no_trabalho',
    #'profissao',
    #'ocupacao',
    #'profissao_companheiro',
    #'grau_instrucao_companheiro',
    #'local_onde_reside',
    #'local_onde_trabalha',
    #'sexo_F',
    #'sexo_M',
    #'sexo_N',
    #'regiao_centro-oeste',
    #'regiao_nordeste',
    #'regiao_norte',
    #'regiao_sudeste',
    #'regiao_sul'
]

# Ajustando escala

ajustador_de_escala = MinMaxScaler()
conjunto_treino_df = conjunto_treino_df[atributos_selecionados]

conjunto_treino_df = ajustador_de_escala.fit_transform(conjunto_treino_df)

conjunto_treino_df = pd.DataFrame(
    conjunto_treino_df,
    columns=atributos_selecionados
)

print('\nVerificar o valor médio de cada atributo em cada classe:\n')

print(conjunto_treino_df.groupby(['inadimplente']).mean().T)

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


# figura = plt.figure(figsize=(15, 12))

# # criar um grafico 3D dentro da figura

# grafico = figura.add_subplot(111, projection='3d')

# grafico.scatter(
#     atributos['renda_mensal_regular'],
#     atributos['idade'],
#     atributos['meses_na_residencia'],
#     c=cores,
#     marker='o',
#     s=10,
#     alpha=1.0
# )


for ax in scatter_matrix.ravel():
    ax.set_xlabel(ax.get_xlabel(), fontsize=7)
    ax.set_ylabel(ax.get_ylabel(), fontsize=7)

dados_treino = conjunto_treino_df.loc[
    :, conjunto_treino_df.columns != 'inadimplente'].values

dados_alvo = conjunto_treino_df.loc[
    :, conjunto_treino_df.columns == 'inadimplente'].values

dados_alvo = dados_alvo.ravel()


# dados_treino, dados_teste, resposta_treino, resposta_teste = train_test_split(
#     dados_treino,
#     dados_alvo,
#     train_size=0.7
# )


# classificador = KNeighborsClassifier(
#     n_neighbors=3,
#     p=2,
#     weights='distance'
# )

# classificador.fit(dados_treino, resposta_treino)

# resposta = classificador.predict(dados_teste)

# total = len(dados_teste)
# acertos = sum(resposta == resposta_teste)
# erros = sum(resposta != resposta_teste)

# print("Total de amostras: ", total)
# print("Respostas corretas:", acertos)
# print("Respostas erradas: ", erros)

for k in range(1, 50, 2):

    classificador = KNeighborsClassifier(
        n_neighbors=k,
        weights='uniform',
        p=1
    )

    scores = cross_val_score(
        classificador,
        dados_treino,
        dados_alvo,
        cv=5
    )

    print(
        'k = ' + str(k),
        'scores =', scores,
        'acurácia média = %6.1f' % (100*sum(scores)/5)
    )


#plt.show()

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