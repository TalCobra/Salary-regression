# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:47:49 2024

@author: Matheus Henrique Dizaro Miyamoto
"""

#%%Instalação pacotes

!pip install pandas
!pip install numpy
!pip install -U seaborn
!pip install matplotlib
!pip install plotly
!pip install scipy
!pip install statsmodels
!pip install scikit-learn
!pip install pingouin
!pip install statstests

#%%Importando os pacotes

import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
import plotly.graph_objects as go # gráficos 3D
from scipy.stats import pearsonr # correlações de Pearson
import statsmodels.api as sm # estimação de modelos
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
from sklearn.preprocessing import LabelEncoder # transformação de dados
import pingouin as pg # outro modo para obtenção de matrizes de correlações
from statstests.process import stepwise # procedimento Stepwise
from statstests.tests import shapiro_francia # teste de Shapiro-Francia
from scipy.stats import boxcox # transformação de Box-Cox
from scipy.stats import norm # para plotagem da curva normal
from scipy import stats # utilizado na definição da função 'breusch_pagan_test'

#%%

salario_dt = pd.read_csv('Salary_dataset.csv',delimiter=',')
#https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression
salario_dt.info()
salario_dt.describe()

salario_dt.rcorr()
#%% Grafico de disperção

plt.figure(figsize=(15, 10))
plt.scatter(x="Salary", y="YearsExperience", data=salario_dt, color='green',
            s=400, label='Valores Reais', alpha=0.7,linewidth=0.2)
plt.title('Dispersão dos dados', fontsize=20)
plt.xlabel('Salário', fontsize=17)
plt.ylabel('Anos de Experiência', fontsize=17)
plt.legend(loc='lower right', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

#%% Montando o modelo

modelo_linear = sm.OLS.from_formula('Salary ~ YearsExperience', salario_dt).fit()
modelo_linear.summary()
#%% Teste Shapiro francia

teste_sf = shapiro_francia(modelo_linear.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() 
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')
#%% Predições

modelo_linear.predict(pd.DataFrame({'YearsExperience':[2.5]}))
