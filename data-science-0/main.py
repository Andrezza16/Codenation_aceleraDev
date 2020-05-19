#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday.head(7)


# In[4]:


index=black_friday.index


# In[5]:


len(index)


# In[6]:


type(index)


# In[7]:


index


# In[8]:


black_friday.columns


# In[9]:


len(black_friday.columns)


# In[10]:


linha=len(black_friday.index)
coluna=len(black_friday.columns)


# In[11]:


linha


# In[12]:


coluna


# In[13]:


black_friday.info()


# In[14]:


black_friday.describe()


# In[15]:


black_friday.sample(20)


# In[16]:


black_friday.tail(10)


# In[17]:


black_friday.groupby('Gender').agg({'Age':'unique'})


# In[18]:


aux=black_friday[(black_friday['Gender']=='F')& (black_friday['Age']=='26-35')]


# In[19]:


aux.shape


# In[20]:


aux.shape[0]


# In[21]:


black_friday[(black_friday['Gender']=='F')& (black_friday['Age']=='26-35')].shape[0]


# In[22]:


resposta3=black_friday['User_ID'].nunique()


# In[23]:


resposta3


# In[24]:


black_friday.dtypes


# In[25]:


resposta4=black_friday.dtypes.nunique()


# In[26]:


resposta4


# In[27]:


black_friday.isnull().sum(axis=0).max()


# In[28]:


valores_nulos=black_friday[black_friday.isnull().any(axis=1)]


# In[29]:


valores_nulos


# In[30]:


nulos2=black_friday[black_friday['Product_Category_2'].isnull()]


# In[31]:


nulos2


# In[32]:


nulos2['Product_Category_3'].isnull().sum()


# In[33]:


a=black_friday.isnull().any(axis = 1).sum()


# In[34]:


a


# In[35]:


b=len(black_friday.index)


# In[36]:


b


# In[37]:


resposta5=a/b


# In[38]:


resposta5


# In[39]:


resposta7=black_friday.Product_Category_3.mode()[0]


# In[40]:


resposta7


# In[41]:


import matplotlib.pyplot as plt


# In[42]:


plt.hist(black_friday.Product_Category_3)


# In[43]:


from sklearn import preprocessing


# In[44]:


nosso_normalizador = preprocessing.MinMaxScaler()


# In[45]:


x=black_friday['Purchase'].values


# In[46]:


x


# In[47]:


r=np.reshape(x,(-1,1))


# In[48]:


dados_normalizados=nosso_normalizador.fit_transform(r)


# In[49]:


dados_normalizados


# In[50]:


resposta8=dados_normalizados.mean()


# In[51]:


resposta8


# In[52]:


scaler = preprocessing.StandardScaler()


# In[53]:


nosso_padronizador = scaler.fit_transform(r)


# In[54]:


nosso_padronizador


# In[55]:


desafio=pd.DataFrame(nosso_padronizador)


# In[56]:


desafio


# In[57]:


desafio.columns=['Valores']


# In[58]:


resposta9=desafio[(desafio['Valores']>-1)&(desafio['Valores']<1)].shape[0]


# In[59]:


resposta9


# In[60]:


resposta10=True


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[61]:


black_friday.shape


# In[62]:


def q1():
    black_friday.shape
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape
    pass


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[63]:


def q2():
    aux.shape[0]
    # Retorne aqui o resultado da questão 2.
    return aux.shape[0]
    pass


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[64]:


def q3():
    resposta3
    # Retorne aqui o resultado da questão 3.
    return resposta3
    pass


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[65]:


def q4():
    resposta4
    # Retorne aqui o resultado da questão 4.
    return resposta4
    pass


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[66]:


def q5():
    # Retorne aqui o resultado da questão 5.
    q5=(len(black_friday) - len(black_friday.dropna())) / len(black_friday)
    
    return q5


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[68]:


def q6():
    black_friday.isnull().sum(axis=0).max()
    # Retorne aqui o resultado da questão 6.
    return black_friday.isnull().sum(axis=0).max()
    pass


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[69]:


def q7():  
    resposta7
    # Retorne aqui o resultado da questão 7.
    return resposta7
    pass


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[79]:


def q8():
    # normalizando:
    black_friday['Purchase_normalizado']=(black_friday['Purchase']-black_friday['Purchase'].min())/(black_friday['Purchase'].max()-black_friday['Purchase'].min())
    
    # calculando a média
    q8=black_friday['Purchase_normalizado'].mean()
    
    return q8


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[71]:


def q9():
    resposta9
    # Retorne aqui o resultado da questão 9.
    return resposta9
    pass


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[72]:


def q10():
    resposta10
    # Retorne aqui o resultado da questão 10.
    return resposta10
    pass

