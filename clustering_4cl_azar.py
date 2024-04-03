import pandas as pd
import numpy as np
import random
from scipy import stats
import pickle
import ast
import matplotlib.pyplot as plt

#%% COHEN D
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

#%% ETIQUETAS, LDA Y MEDIANA

# lda = pd.read_csv('C:/Users/xochipilli/Documents/things-eeg/junio/prueba_grande/lda_df.csv')
# mediana = pd.read_csv('C:/Users/xochipilli/Documents/things-eeg/junio/prueba_grande/mediana_df.csv')
lda = pd.read_csv('/home/usuario/Escritorio/Eric/prueba_grande/lda_df.csv')
mediana = pd.read_csv('/home/usuario/Escritorio/Eric/prueba_grande/mediana_df.csv')

# with open('C:/Users/xochipilli/Documents/things-eeg/scripts/clustering/etiquetas.txt', 'r') as f:
with open('/home/usuario/Escritorio/Eric/scripts/clustering/etiquetas.txt', 'r') as f:
    archivo = f.read()
etiquetas = ast.literal_eval(archivo)

#%% COHEN y SPEARMAN PARA MATRICES DE CLUSTERS
# num = 0
# print('NUM {}'.format(num))

f_dic = '/home/usuario/Escritorio/Eric/scripts/clustering/lum_'
# f_dic = 'C:/Users/xochipilli/Documents/things-eeg/scripts/clustering/lum_'
with open(f_dic + 'lda_cohen.pkl', 'rb') as f:
      lda_cohen_ref = pickle.load(f)
with open(f_dic + 'med_cohen.pkl', 'rb') as f:
      med_cohen_ref = pickle.load(f)
with open(f_dic + 'lda_spearman.pkl', 'rb') as f:
      lda_spearman_ref = pickle.load(f)
with open(f_dic + 'med_spearman.pkl', 'rb') as f:
      med_spearman_ref = pickle.load(f)

cant0, cant1, cant2, cant3, cantdis = 0, 0, 0, 0, 0
for i in range(1853):
    for j in range(i+1,1854):
        if etiquetas[i] == etiquetas[j] == 0:
            cant0 += 1
        elif etiquetas[i] == etiquetas[j] == 1: 
            cant1 += 1
        elif etiquetas[i] == etiquetas[j] == 2: 
            cant2 += 1
        elif etiquetas[i] == etiquetas[j] == 3: 
            cant3 += 1
        else:
            cantdis += 1

str_i = [str(i) for i in range(275)]
clusters = [0,1,2,3,-1]

azar_lda_cohen, azar_med_cohen = {}, {}
# lum_lda_spearman, lum_med_spearman = {}, {}
# dics = [lum_lda_cohen, lum_med_cohen]
for d in [azar_lda_cohen, azar_med_cohen]:
    for c in [0,1,2,3]:
        d[c] = np.zeros(275)
        
# folder = 'C:/Users/xochipilli/Documents/things-eeg/scripts/clustering/etiq_falsas/{}/'.format(num)
folder = '/home/usuario/Escritorio/Eric/scripts/clustering/etiq_falsas/'

azar = (list(range(1717731)))
for j in range(100):
    
    random.shuffle(azar)
    coincidencias = {}
    coincidencias[0] = set(azar[:cant0])
    coincidencias[1] = set(azar[cant0:cant0+cant1])
    coincidencias[2] = set(azar[cant0+cant1:cant0+cant1+cant2])
    coincidencias[3] = set(azar[cant0+cant1+cant2:cant0+cant1+cant2+cant3])
    # coincidencias[-1] = set(azar[-cantdis:])
    
    for c in [0,1,2,3]:
        a = coincidencias[c] #coincidencias
        b = list(set(azar) - a)
        # aux = [coincidencias[s] for s in clusters if s!= c]
        # b = [x for _ in aux for x in _] #disidencias
        cohenlda = [cohend(lda[i][b], lda[i][list(a)]) for i in str_i]   #disidencias - coincidencias
        cohenmed = [cohend(mediana[i][b], mediana[i][list(a)]) for i in str_i]
        # cero_spear = np.zeros(1717731)
        # for _ in a:
        #     cero_spear[_] = 1
        # rholda = [stats.spearmanr(cero_spear, lda[i])[0] for i in str_i]
        # rhomed = [stats.spearmanr(cero_spear, mediana[i])[0] for i in str_i]
        
        azar_lda_cohen[c] += [1 if abs(cohenlda[i])<abs(lda_cohen_ref[c][i]) else 0 for i in range(275)]
        azar_med_cohen[c] += [1 if abs(cohenmed[i])<abs(med_cohen_ref[c][i]) else 0 for i in range(275)]
        # lum_lda_spearman[c] += [1 if rholda[i]>lda_spearman_ref[c][i] else 0 for i in range(275)]
        # lum_med_spearman[c] +=  [1 if rhomed[i]>med_spearman_ref[c][i] else 0 for i in range(275)]
    # print('Corrida {}'.format(j))
        
    # df_lum_lda_cohen = pd.DataFrame(lum_lda_cohen)
    # df_lum_lda_cohen.to_csv('{}lda_cohen.csv'.format(folder), index=False)
    # df_lum_med_cohen = pd.DataFrame(lum_med_cohen)
    # df_lum_med_cohen.to_csv('{}med_cohen.csv'.format(folder), index=False)
    
    # df_lum_lda_rho = pd.DataFrame(lum_lda_spearman)
    # df_lum_lda_rho.to_csv('{}lda_spearman.csv'.format(folder), index=False)
    # df_lum_med_rho = pd.DataFrame(lum_med_spearman)
    # df_lum_med_rho.to_csv('{}med_spearman.csv'.format(folder), index=False)
    
dic_azar = {}
for t in ['lda', 'med']:
    dic_azar[t] = {}
    for c in [0,1,2,3]:
         azar = [1 if clusters_cohen_lda_med[t][c][j] > 94 else 0 for j in range(275)]
         dic_azar[t][c] = azar
     
# with open(folder + 'clusters_cohen_lda_med_azar.pkl', 'wb') as f:
#       pickle.dump(dic_azar, f) 
      
      
#%% CARGO LOS DATOS ORIGINALES Y ETIQUETAS ALEATORIAS
# f_dic = 'C:/Users/xochipilli/Documents/things-eeg/scripts/clustering/'
f_dic = '/home/usuario/Escritorio/Eric/scripts/clustering/'

with open(f_dic + 'lum_lda_cohen.pkl', 'rb') as f:
    lum_lda_cohen = pickle.load(f)
with open(f_dic + 'lum_med_cohen.pkl', 'rb') as f:
    lum_med_cohen = pickle.load(f)
with open(f_dic + 'lum_lda_spearman.pkl', 'rb') as f:
    lum_lda_spearman = pickle.load(f)
with open(f_dic + 'lum_med_spearman.pkl', 'rb') as f:
    lum_med_spearman = pickle.load(f)
with open(f_dic + 'clusters_cohen_lda_med_azar.pkl', 'rb') as f:
    clusters_cohen_lda_med = pickle.load(f)
