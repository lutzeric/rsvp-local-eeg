import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#%% DATASETS
# lda = pd.read_csv('C:/Users/xochipilli/Documents/things-eeg/junio/prueba_grande/lda_df.csv')
# mediana = pd.read_csv('C:/Users/xochipilli/Documents/things-eeg/junio/prueba_grande/mediana_df.csv')
# X = pd.read_csv('C:/Users/xochipilli/Documents/things-eeg/junio/luminancia.csv')
lda = pd.read_csv('/home/ubuntuverde/Eric/rsvp/prueba_grande/lda_df.csv')
mediana = pd.read_csv('/home/ubuntuverde/Eric/rsvp/prueba_grande/mediana_df.csv')
X = pd.read_csv('/home/ubuntuverde/Eric/rsvp/scripts/luminancia.csv')

X.drop(columns=['item'], inplace=True)
X = X.groupby(['concepto']).mean()

#%% DIMENSIONES FINALES
#definitivas:
saco = ['RMS', 'oriEn_1', 'oriEn_10', 'oriEn_11', 'oriEn_13', 'oriEn_15', 
        'oriEn_16', 'oriEn_17', 'oriEn_2', 'oriEn_3', 'oriEn_5', 'oriEn_7', 
        'oriEn_8', 'oriEn_9', 'sfEn_2', 'sfEn_3', 'sfEn_4', 'sfEn_8',
        'oriEn_4', 'r', 'oriEn_12', 'sfEn_1']

X2 = X.drop(columns=saco)

std_scale = StandardScaler()
std_scale.fit(X2)
X_scaled = std_scale.transform(X2)
pca = PCA(n_components=2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

#values
vals = np.around(pca.components_**2,3)

cp1 = [p[0] for p in X_pca]
cp2 = [p[1] for p in X_pca]

dif_1 = np.zeros(1717731)
dif_2 = np.zeros(1717731)

k = 0
for i in range(1853):
    for j in range(i+1,1854):
        dif_1[k] = abs(cp1[i] - cp1[j])
        dif_2[k] = abs(cp2[i] - cp2[j])
        k += 1

#%% GRAFICOS
x = range(275)

dic_rho_p = {}
comb_rho = [[scipy.stats.spearmanr(dif_1, lda[str(i)])[0] for i in x],
            [scipy.stats.spearmanr(dif_1, mediana[str(i)])[0] for i in x],
            [scipy.stats.spearmanr(dif_2, lda[str(i)])[0] for i in x],
            [scipy.stats.spearmanr(dif_2, mediana[str(i)])[0] for i in x]]

comb_p = [[scipy.stats.spearmanr(dif_1, lda[str(i)])[1] for i in x],
            [scipy.stats.spearmanr(dif_1, mediana[str(i)])[1] for i in x],
            [scipy.stats.spearmanr(dif_2, lda[str(i)])[1] for i in x],
            [scipy.stats.spearmanr(dif_2, mediana[str(i)])[1] for i in x]]

titulos = ['LDA 1CP', 'Mediana 1CP', 'LDA 2CP', 'Mediana 2CP']

# path = 'C:/Users/xochipilli/Documents/things-eeg/scripts/pca/'
# with open(path + 'dic_rho_p.pkl', 'wb') as f:
#       pickle.dump(dic_rho_p, f)  
      
#%%
path = '/home/usuario/Escritorio/Eric/scripts/pca/'
with open(path + 'dic_rho_p.pkl', 'rb') as f:
    rho = pickle.load(f)    

#%% CORRELACION DIF CP VS DISTANCIA SEMANTICA
word = pd.read_csv('C:/Users/xochipilli/Documents/things-eeg/junio/df_combinaciones_word2vec.csv')
giga = word['gigaword'].loc[word['gigaword']!=1]
ind_giga = list(giga.index)
wiki = word['wikipedia'].loc[word['wikipedia']!=1]
ind_wiki = list(wiki.index)

ind = [ind_wiki, ind_giga]
base = [wiki, giga]
nombre = ['Wikipedia', 'Gigaword']
pc = ['segundo', 'primer']
dif = [dif_2, dif_1]

for i in range(2):
    r = scipy.stats.spearmanr(dif[i][ind[i]], base[i])[0]
    sns.regplot(dif[i][ind[i]], base[i])
    plt.xlabel('Diferencia en el componente principal')
    plt.ylabel('Distancia semántica')
    plt.title('Diferencia en el {} CP vs distancia según {}. R='.format(pc[i], nombre[i]) + str(round(r,3)))
    plt.show()
