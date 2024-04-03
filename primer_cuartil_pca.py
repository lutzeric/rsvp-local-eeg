import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
import pickle
import pingouin as pg

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

#%% DATASETS LUMINANCIA, LDA Y MEDIANA

# lda = pd.read_csv('C:/Users/xochipilli/Documents/things-eeg/junio/prueba_grande/lda_df.csv')
# mediana = pd.read_csv('C:/Users/xochipilli/Documents/things-eeg/junio/prueba_grande/mediana_df.csv')
lda = pd.read_csv('/home/usuario/Escritorio/Eric/prueba_grande/lda_df.csv')
mediana = pd.read_csv('/home/usuario/Escritorio/Eric/prueba_grande/mediana_df.csv')

lda = lda.to_numpy()
mediana = mediana.to_numpy()

# dfs = [lda,mediana]

# X = pd.read_csv('C:/Users/xochipilli/Documents/things-eeg/junio/luminancia.csv')
X = pd.read_csv('/home/usuario/Escritorio/Eric/scripts/luminancia.csv')
X.drop(columns=['item', 'oriEn_17'], inplace=True)
X = X.groupby(['concepto']).mean()

#%% ENCUENTRO CANTIDAD DE COINCIDENCIAS PARA CADA PRIMER CUARTIL DE CADA COMPONENTE

std_scale = StandardScaler()
std_scale.fit(X)
X_scaled = std_scale.transform(X)
pca = PCA(n_components=2)
pca.fit(X_scaled)

# componentes = [0,1,2,3,4]
X_pca = pca.transform(X_scaled)
X_pca = X_pca.T

#son 465 por componente
#en total 465*464/2 coincidencias = 107880
    
# path2 = '/home/usuario/Escritorio/Eric/scripts/mediana/'
sujetos = set(range(1,51)) - {6}

# with open(path2 + 'dic_lda.pkl', 'rb') as f:    #prueba grande
#     lda = pickle.load(f)  

# with open(path2 + 'dic_mediana.pkl', 'rb') as f:    #prueba grande
#     mediana = pickle.load(f)  
    
#%% COHEN Y SPEARMAN del PRIMER CUARTIL
ind_ord0 = np.argsort(abs(X_pca[0]))[-464:]  #primer cuartil, primer CP
ind_ord1 = np.argsort(abs(X_pca[1]))[-464:]  #primer cuartil, segundo CP

coin0, dis0 = [], []
coin1, dis1 = [], []

k = 0
for i in range(1853):
    for j in range(i+1, 1854):
        if i in ind_ord0 and j in ind_ord0:
            coin0.append(k)
        else:
            dis0.append(k)
            
        if i in ind_ord1 and j in ind_ord1:
            coin1.append(k)
        else:
            dis1.append(k)
        k += 1
        
lum_lda_cohen_cuartil = {0:np.zeros(275), 1:np.zeros(275)}
lum_med_cohen_cuartil = {0:np.zeros(275), 1:np.zeros(275)}

for t in range(275):
    lum_lda_cohen_cuartil[0][t] = cohend(lda[coin0,t], lda[dis0,t])
    lum_lda_cohen_cuartil[1][t] = cohend(lda[coin1,t], lda[dis1,t])
    lum_med_cohen_cuartil[0][t] = cohend(mediana[coin0,t], mediana[dis0,t])
    lum_med_cohen_cuartil[1][t] = cohend(mediana[coin1,t], mediana[dis1,t])
    
carpeta = '/home/usuario/Escritorio/Eric/prueba_grande/cuartil_pca/'
        
cero_spear = np.zeros(1717731)
cero_spear[dis0] = 1
uno_spear = np.zeros(1717731)
uno_spear[dis1] = 1
    
lum_lda_spearman_cuartil, lum_med_spearman_cuartil = {}, {}
lum_lda_spearman_cuartil[0] = [scipy.stats.spearmanr(cero_spear, lda[:,i]) for i in range(275)]
lum_med_spearman_cuartil[0] = [scipy.stats.spearmanr(cero_spear, mediana[:,i]) for i in range(275)]
lum_lda_spearman_cuartil[1] = [scipy.stats.spearmanr(uno_spear, lda[:,i]) for i in range(275)]
lum_med_spearman_cuartil[1] = [scipy.stats.spearmanr(uno_spear, mediana[:,i]) for i in range(275)]

with open(carpeta + 'lum_lda_spearman_cuartil.pkl', 'wb') as f:
    pickle.dump(lum_lda_spearman_cuartil, f)
with open(carpeta + 'lum_med_spearman_cuartil.pkl', 'wb') as f:
    pickle.dump(lum_med_spearman_cuartil, f)

spearman_lda_med = [scipy.stats.spearmanr(lda[t],mediana[t])[0] for t in range(275)]
with open(carpeta + 'lda_med_spearman.pkl', 'wb') as f:
    pickle.dump(spearman_lda_med, f)
    
#%%
# x = range(-100,1000,4)
# low,up = pg.compute_esci(stat=lum_lda_cohen_cuartil[0], nx=107416, ny=1610315, eftype='cohen', decimals=3)
# plt.plot(x,lum_lda_cohen_cuartil[0], color='red')
# plt.fill_between(x, low, up, color='red', alpha=0.2)
# low,up = pg.compute_esci(stat=lum_med_cohen_cuartil[0], nx=107416, ny=1610315, eftype='cohen', decimals=3)
# plt.plot(x,lum_med_cohen_cuartil[0], color='blue')
# plt.fill_between(x, low, up, color='blue', alpha=0.2)
# plt.show()

# low,up = pg.compute_esci(stat=lum_lda_cohen_cuartil[1], nx=107416, ny=1610315, eftype='cohen', decimals=3)
# plt.plot(x,lum_lda_cohen_cuartil[1], color='red')
# plt.fill_between(x, low, up, color='red', alpha=0.2)
# low,up = pg.compute_esci(stat=lum_med_cohen_cuartil[1], nx=107416, ny=1610315, eftype='cohen', decimals=3)
# plt.plot(x,lum_med_cohen_cuartil[1], color='blue')
# plt.fill_between(x, low, up, color='blue', alpha=0.2)
# plt.show()

#%% AL AZAR
azar_lda_cohen, azar_med_cohen = {}, {}
azar_lda_spearman, azar_med_spearman = {}, {}
str_i = list(range(275))

for c in [0,1]:
    azar_lda_cohen[c], azar_med_cohen[c] = np.zeros((100,275)), np.zeros((100,275))
    azar = (list(range(1717731)))
    
    for k in range(100):
        random.shuffle(azar)
        a = azar[:107416] #coincidencias
        b = azar[107416:] #disidencias
                
        azar_lda_cohen[c][k] = [cohend(lda[:,i][b], lda[:,i][a]) for i in str_i]   #disidencias - coincidencias
        azar_med_cohen[c][k] = [cohend(mediana[:,i][b], mediana[:,i][a]) for i in str_i]
        
lum_lda_cohen_cuartil['azar'] = azar_lda_cohen
lum_med_cohen_cuartil['azar'] = azar_med_cohen
   
sig_lda_0 = [np.sum([abs(lum_lda_cohen_cuartil[0][t]) > abs(lum_lda_cohen_cuartil['azar'][0][n][t]) for n in range(100)])>95 for t in range(275)]
sig_lda_1 = [np.sum([abs(lum_lda_cohen_cuartil[1][t]) > abs(lum_lda_cohen_cuartil['azar'][1][n][t]) for n in range(100)])>95 for t in range(275)]
sig_med_0 = [np.sum([abs(lum_med_cohen_cuartil[0][t]) > abs(lum_med_cohen_cuartil['azar'][0][n][t]) for n in range(100)])>95 for t in range(275)]
sig_med_1 = [np.sum([abs(lum_med_cohen_cuartil[1][t]) > abs(lum_med_cohen_cuartil['azar'][1][n][t]) for n in range(100)])>95 for t in range(275)]

lum_lda_cohen_cuartil['sig0'] = sig_lda_0
lum_lda_cohen_cuartil['sig1'] = sig_lda_1
lum_med_cohen_cuartil['sig0'] = sig_med_0
lum_med_cohen_cuartil['sig1'] = sig_med_1

with open(carpeta + 'lum_lda_cohen_cuartil.pkl', 'wb') as f:
    pickle.dump(lum_lda_cohen_cuartil, f)
with open(carpeta + 'lum_med_cohen_cuartil.pkl', 'wb') as f:
    pickle.dump(lum_med_cohen_cuartil, f)
    
#%% IMAGENES REPRESENTATIVAS
conceptos = X.index

#primer CP
print(list(conceptos[ind_ord0[-5:]]))
print(list(conceptos[ind_ord1[-5:]]))

#cp1: ['spider web', 'fireworks', 'doily', 'porcupine', 'sticker']
#cp2: ['fire', 'tick', 'bow2', 'footprint', 'sand']
