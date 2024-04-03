import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import ast
import pickle
import copy
import random

#%% FUNCIONES
# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = np.mean(d1), np.mean(d2)
	# calculate the effect size
	return (u1 - u2) / s

def cargar_df(ruta, plot=False):
    df = pd.read_csv(ruta)
    if plot==True:
        for c in df.columns:
            plt.plot(range(275), df[c])
            plt.title(c)
            plt.show()
    return df

def matriz(cat_con, con_cat, masgrandes):
    matrix = []
    for m in masgrandes:
        vector = np.zeros(len(masgrandes))
        for n in range(len(masgrandes)):
            if m != masgrandes[n]:
                for concepto in cat_con[m]:
                    if masgrandes[n] in con_cat[concepto]:
                        vector[n] += 1
            else:
                vector[n] = len(cat_con[m])
        # print(m, len(cat_con[m]))
        # print(vector)
        vector /= len(cat_con[m])
        matrix.append(vector*100)
    return matrix

#%% DICCIONARIO DE CADA CATEGORIA y CONCEPTO
# f_dic = 'C:/Users/xochipilli/Documents/things-eeg/junio/prueba_grande/nuevos/cohen_rho/categorias/'
f_dic = '/home/usuario/Escritorio/Eric/scripts/categorias/'
# temp = pd.read_csv(f_dic + 'categorias_conceptos.csv')

with open(f_dic + 'cat_con.pkl', 'rb') as f:
      cat_con = pickle.load(f)
    
with open(f_dic + 'con_cat.pkl', 'rb') as f:
      con_cat = pickle.load(f)
     

list_cat = [cat_con[k] for k in cat_con.keys()]
# conceptos = set([l for c in list_cat for l in c])
# con_cat = {c:[cat for cat in cat_con.keys() if c in cat_con[cat]] for c in conceptos}
categorias = list(cat_con.keys())
conceptos = list(con_cat.keys())
conceptos.sort()

# with open(f_dic + 'cat_con.pkl', 'wb') as f:
#     pickle.dump(cat_con, f)

# with open(f_dic + 'con_cat.pkl', 'wb') as f:
#     pickle.dump(con_cat, f)
      
#define dimensions of subplots (rows, columns)

# con_cat2 = {temp['concepto'][i]:ast.literal_eval(temp['otra'][i]) for i in range(1854)}
# cat_con2 = {k:[o for o in con_cat2.keys() if k in con_cat2[o]] for k in categorias}

#%% CARGO LAS MATRICES y LOS DATASETS DE LDA Y MEDIANA

# lda = pd.read_csv('C:/Users/xochipilli/Documents/things-eeg/junio/prueba_grande/lda_df.csv')
# mediana = pd.read_csv('C:/Users/xochipilli/Documents/things-eeg/junio/prueba_grande/mediana_df.csv')
lda = pd.read_csv('/home/usuario/Escritorio/Eric/prueba_grande/lda_df.csv')
mediana = pd.read_csv('/home/usuario/Escritorio/Eric/prueba_grande/mediana_df.csv')

lda = np.array(lda)
mediana = np.array(mediana)

#%% COHEN Y D PARA MATRICES DE CATEGORIAS
coin_cat, dis_cat = {}, {} #indices para categorias con mas de 18 coincidencias
# validas = []
validas2 = ['human', 'body', 'animal', 'hardware']

# for c in categorias:
for c in validas2:
    if len(cat_con[c]) > 18:    #si tiene más del 1% de los conceptos
        k = 0
        coin, dis = [], []
        for i in range(1853):
            for j in range(i+1, 1854):
                if conceptos[i] in cat_con[c] and conceptos[j] in cat_con[c]:
                    coin.append(k)
                else:
                    dis.append(k)
                k += 1

        coin_cat[c] = coin #coincidencias
        dis_cat[c] = dis #disidencias
        # validas.append(c)
        
#hay 37 válidas, pero me quedo con las que habían dado la otra vez:

# validas2 = ['clothing', 'body', 'tool', 'natural', 'fruit', 'food', 'musical instrument',
#             'furniture', 'dessert', 'animal', 'drink', 'human']

#%% COHEN Y SPEARMAN PARA LAS VALIDAS
# cat_lda_cohen, cat_med_cohen = {}, {}
cat_lda_spearman, cat_med_spearman = {}, {}

validas2 = ['human', 'body', 'animal', 'hardware']
# contador = 0
for c in validas2:
# for c in ['drink']:
    # print(len(validas)-contador)
    a = coin_cat[c]
    b = dis_cat[c]
    # cat_lda_cohen[c] = [cohend(lda[i][b], lda[i][a]) for i in str_i]   #disidencias - coincidencias
    # cat_med_cohen[c] = [cohend(mediana[i][b], mediana[i][a]) for i in str_i]
    
    cero_spear = np.zeros(1717731)
    cero_spear[a] = 1
    
    cat_lda_spearman[c] = [spearmanr(cero_spear, lda[:,i]) for i in range(275)]     #disidencias - coincidencias
    cat_med_spearman[c] = [spearmanr(cero_spear, mediana[:,i]) for i in range(275)]
    

    # contador += 1

    # with open(f_dic + 'cat_lda_cohen.pkl', 'wb') as f:
    #     pickle.dump(cat_lda_cohen, f)
    # with open(f_dic + 'cat_med_cohen.pkl', 'wb') as f:
    #     pickle.dump(cat_med_cohen, f)
    # with open(f_dic + 'cat_lda_spearman.pkl', 'wb') as f:
    #     pickle.dump(cat_lda_spearman, f)
    # with open(f_dic + 'cat_med_spearman.pkl', 'wb') as f:
    #     pickle.dump(cat_med_spearman, f)
    
#las que parecen dar son hardware, body, tool, natural, food, clothing, animal, human

#%% ETIQUETAS AL AZAR
with open(f_dic + 'cat_lda_cohen.pkl', 'rb') as f:
      cat_lda_cohen = pickle.load(f)
with open(f_dic + 'cat_med_cohen.pkl', 'rb') as f:
      cat_med_cohen = pickle.load(f)

validas2 = ['hardware','human','body','animal','tool','natural']

str_i = [str(i) for i in range(275)]
azar_lda_cohen, azar_med_cohen = {},{}
# with open(f_dic + 'dic_positivos.pkl', 'rb') as f:
#       dic_pos = pickle.load(f)
dic_pos = {'lda':{}, 'med':{}}

contador = 0
for c in validas2:
    print(len(validas2)-contador)
    azar_lda_cohen[c] = np.zeros(275)
    azar_med_cohen[c] = np.zeros(275)
    for _ in range(100):
        coin_real = len(cat_con[c])
        lista = random.sample(range(1854),1854)
        coin, dis = [], []
        
        a = lista[:coin_real]   #coincidencias al azar
        b = lista[coin_real:]   #disidencias al azar
        azar_lda = [cohend(lda[i][b], lda[i][a]) for i in str_i]  #disidencias - coincidencias
        azar_med = [cohend(mediana[i][b], mediana[i][a]) for i in str_i]
        azar_lda_cohen[c] += [1 if abs(azar_lda[r])<abs(cat_lda_cohen[c][r]) else 0 for r in range(275)]
        azar_med_cohen[c] += [1 if abs(azar_med[r])<abs(cat_med_cohen[c][r]) else 0 for r in range(275)]

    cont_lda = [1 if azar_lda_cohen[c][r]>95 else 0 for r in range(275)]
    cont_med = [1 if azar_med_cohen[c][r]>95 else 0 for r in range(275)]
    
    dic_pos['lda'][c] = cont_lda
    dic_pos['med'][c] = cont_med
    
    with open(f_dic + 'dic_positivos.pkl', 'wb') as f:
        pickle.dump(dic_pos, f)
        
    contador += 1