import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as shc
import ast
from sklearn.neighbors import NearestCentroid
# from sklearn.cluster import KMeans
import pickle
from matplotlib.ticker import FormatStrFormatter

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
# path = 'C:/Users/xochipilli/Documents/things-eeg/scripts/mediana/'

# with open(path + 'dic_mediana.pkl', 'rb') as f:
#     dic_mediana = pickle.load(f)   
# mediana = dic_mediana['media']  
lda = pd.read_csv('/home/usuario/Escritorio/Eric/prueba_grande/lda_df.csv')
mediana = pd.read_csv('/home/usuario/Escritorio/Eric/prueba_grande/mediana_df.csv')
X = pd.read_csv('/home/usuario/Escritorio/Eric/scripts/luminancia.csv')

# lda = pd.read_csv('C:/Users/xochipilli/Documents/things-eeg/junio/prueba_grande/lda_df.csv')
# mediana = pd.read_csv('C:/Users/xochipilli/Documents/things-eeg/junio/prueba_grande/mediana_df.csv')
# X = pd.read_csv('C:/Users/xochipilli/Documents/things-eeg/junio/luminancia.csv')

saco = ['RMS', 'oriEn_1', 'oriEn_10', 'oriEn_11', 'oriEn_13', 'oriEn_15', 
        'oriEn_16', 'oriEn_17', 'oriEn_2', 'oriEn_3', 'oriEn_5', 'oriEn_7', 
        'oriEn_8', 'oriEn_9', 'sfEn_2', 'sfEn_3', 'sfEn_4', 'sfEn_8',
        'oriEn_4', 'r', 'oriEn_12', 'sfEn_1']

X.drop(columns=saco, inplace=True)
X2 = X.groupby(['concepto']).mean()

#%% CLUSTERS JERARQUICOS
std_scale = StandardScaler()
std_scale.fit(X2)
X_scaled = std_scale.transform(X2)

plt.figure(figsize=(10, 7))
plt.title("Dendrograma de objetos por características de la luz")
plt.ylabel("Distancia ward")
dend = shc.dendrogram(shc.linkage(X_scaled, method='ward'))
# se ven 4 clusters y el método del codo sugiere 4

# from kneed import KneeLocator # importamos el paquete para detectar el codo
# sse = [] # acá vamos a guardar el puntaje de la función objetivo
# for k in range(1, 20):
#   kkmeans = KMeans(n_clusters=k)
#   kkmeans.fit(X2)
#   sse.append(kkmeans.inertia_)
# kl = KneeLocator(range(1, 20), sse, curve="convex", direction="decreasing")
# print("El codo está en k =", kl.elbow)

cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
etiquetas = cluster.fit_predict(X_scaled)

c0 = np.count_nonzero(etiquetas == 0)
c1 = np.count_nonzero(etiquetas == 1)
c2 = np.count_nonzero(etiquetas == 2)
c3 = np.count_nonzero(etiquetas == 3)
print(c0, c1, c2, c3)

# with open('C:/Users/xochipilli/Documents/things-eeg/scripts/clustering/etiquetas.txt', 'w') as f:
#     f.write(str(list(etiquetas)))

'''da mejor cluster aglomerativo que K means'''
# kmeans = KMeans(n_clusters=4, random_state=0).fit(X_scaled)
# labels = kmeans.labels_
# kmeans.cluster_centers_

#da mejor con clustering aglomerativo que con Kmeans
    
#%% RADAR PLOT
clf = NearestCentroid()
clf.fit(X_scaled, etiquetas)
clf.centroids_

df = pd.DataFrame(dict(
    Cluster0=clf.centroids_[0], #kmeans.cluster_centers_[0],
    theta=X2.columns))

df['Cluster1'] = clf.centroids_[1] #kmeans.cluster_centers_[1] #
df['Cluster2'] = clf.centroids_[2] #kmeans.cluster_centers_[2] #
df = df.T
df.drop(['theta'], inplace=True)
df.columns = X2.columns
# df['lum'] = -df['lum']
df = df.astype('float')
# df2 = np.log(df)

#%% COHEN y SPEARMAN PARA MATRICES DE CLUSTERS
#probar con etiquetas y con kmeans.labels_
# etiquetas = kmeans.labels_
coin_0, coin_1, coin_2, coin_3, dis = [], [], [], [], []
k = 0
for i in range(1853):
    for j in range(i+1, 1854):
        if etiquetas[i] == etiquetas[j] == 0:
            coin_0.append(k)
        elif etiquetas[i] == etiquetas[j] == 1:
            coin_1.append(k)
        elif etiquetas[i] == etiquetas[j] == 2:
            coin_2.append(k)
        elif etiquetas[i] == etiquetas[j] == 3:
            coin_3.append(k)
        # elif etiquetas[i] == etiquetas[j] == 4:
        #     coin_4.append(k)
        else:
            dis.append(k)
        k += 1
        
# print(0, sum(coin_0))
# print(1, sum(coin_1))
# print(2, sum(coin_2))
# print(3, sum(coin_3))

coincidencias = {}

coincidencias[-1] = dis
coincidencias[0] = coin_0
coincidencias[1] = coin_1
coincidencias[2] = coin_2
coincidencias[3] = coin_3
# coincidencias[4] = coin_4
str_i = [str(i) for i in range(275)]

lum_lda_cohen, lum_med_cohen = {}, {}
lum_lda_spearman, lum_med_spearman = {}, {}
lda_med_spearman = {}
clusters = [0,1,2,3,-1]
for c in [0,1,2,3]:
    a = coincidencias[c] #coincidencias
    # aux = [coincidencias[s] for s in clusters if s!= c]
    # b = [x for _ in aux for x in _] #disidencias
    # lum_lda_cohen[c] = [cohend(lda[i][b], lda[i][a]) for i in str_i]   #disidencias - coincidencias
    # lum_med_cohen[c] = [cohend(mediana[i][b], mediana[i][a]) for i in str_i]
    cero_spear = np.zeros(1717731)
    cero_spear[a] = 1
    lum_lda_spearman[c] = [scipy.stats.spearmanr(cero_spear, lda[i]) for i in str_i]
    lum_med_spearman[c] = [scipy.stats.spearmanr(cero_spear, mediana[i]) for i in str_i]
    lda_med_spearman[c] = [scipy.stats.spearmanr(lda[i], mediana[i]) for i in str_i]
    

# f_dic = 'C:/Users/xochipilli/Documents/things-eeg/scripts/clustering/'
f_dic = '/home/usuario/Escritorio/Eric/scripts/clustering/'

# with open(f_dic + 'lum_lda_cohen.pkl', 'wb') as f:
#     pickle.dump(lum_lda_cohen, f)
# with open(f_dic + 'lum_med_cohen.pkl', 'wb') as f:
#     pickle.dump(lum_med_cohen, f)
# with open(f_dic + 'lum_lda_spearman.pkl', 'wb') as f:
#     pickle.dump(lum_lda_spearman, f)
# with open(f_dic + 'lum_med_spearman.pkl', 'wb') as f:
#     pickle.dump(lum_med_spearman, f)
# with open(f_dic + 'lda_med_spearman.pkl', 'wb') as f:
#     pickle.dump(lda_med_spearman, f)
    
#%% ENCUENTRO LOS INDICES QUE COINCIDEN PARA CADA PRIMER CUARTIL DE CADA COMPONENTE
std_scale = StandardScaler()
std_scale.fit(X2)
X_scaled = std_scale.transform(X2)
pca = PCA(n_components=5)
pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)
X_pca = X_pca.T
comps = {}
for c in [0,1,2,3]:
    b = sorted(enumerate(X_pca[c]), key=lambda i: i[1])
    comps[c] = [b[i][0] for i in range(465)]

compa_pca, dis_pca = {}, {}
compa_pca[0], compa_pca[1], compa_pca[2], compa_pca[3], compa_pca[4] = [], [], [], [], []
dis_pca[0], dis_pca[1], dis_pca[2], dis_pca[3], dis_pca[4] = [], [], [], [], []

for c in [0,1,2,3]:
    k = 0
    for i in range(1853):
        for j in range(i+1, 1854):
            if i in comps[c] and j in comps[c]:
                compa_pca[c].append(k)
            else:
                dis_pca[c].append(k)
            k += 1
    
#%% COHEN Y SPEARMAN del PRIMER CUARTIL
lum_lda_cohen_cuartil, lum_med_cohen_cuartil = {}, {}
lum_lda_spearman_cuartil, lum_med_spearman_cuartil = {}, {}
for c in [0,1,2,3]:
    a = compa_pca[c] #coincidencias
    b = dis_pca[c] #disidencias
    lum_lda_cohen_cuartil[c] = [cohend(lda[i][b], lda[i][a]) for i in str_i]   #disidencias - coincidencias
    lum_med_cohen_cuartil[c] = [cohend(mediana[i][b], mediana[i][a]) for i in str_i]
    cero_spear = np.zeros(1717731)
    for _ in a:
        cero_spear[_] = 1
    lum_lda_spearman_cuartil[c] = [scipy.stats.spearmanr(cero_spear, lda[i])[0] for i in str_i]
    lum_med_spearman_cuartil[c] = [scipy.stats.spearmanr(cero_spear, mediana[i])[0] for i in str_i]
    plt.plot(range(275), lum_lda_cohen_cuartil[c])
    plt.title('Cohen LDA CP {} 1er cuartil'.format(c))
    plt.show()
    plt.plot(range(275), lum_med_cohen_cuartil[c])
    plt.title('Cohen mediana CP {} 1er cuartil'.format(c))
    plt.show()
    plt.plot(range(275), lum_lda_spearman_cuartil[c])
    plt.title('Spearman LDA CP {} 1er cuartil'.format(c))
    plt.show()
    plt.plot(range(275), lum_med_spearman_cuartil[c])
    plt.title('Spearman mediana CP {} 1er cuartil'.format(c))
    plt.show()

# folder = 'C:/Users/xochipilli/Documents/things-eeg/scripts/clustering/'


# df_lum_lda_cohen_cuartil = pd.DataFrame(lum_lda_cohen_cuartil).round(3)
# df_lum_lda_cohen_cuartil.to_csv('{}lda_cohen_cuartil.csv'.format(folder), index=False)
# df_lum_med_cohen_cuartil = pd.DataFrame(lum_med_cohen_cuartil).round(3)
# df_lum_med_cohen_cuartil.to_csv('{}med_cohen_cuartil.csv'.format(folder), index=False)

# df_lum_lda_rho_cuartil = pd.DataFrame(lum_lda_spearman_cuartil).round(3)
# df_lum_lda_rho_cuartil.to_csv('{}lda_spearman_cuartil.csv'.format(folder), index=False)
# df_lum_med_rho_cuartil = pd.DataFrame(lum_med_spearman_cuartil).round(3)
# df_lum_med_rho_cuartil.to_csv('{}med_spearman_cuartil.csv'.format(folder), index=False)

#%% LEVANTO LOS DATOS Y GRAFICO
folder = '/home/usuario/Escritorio/Eric/scripts/clustering/'

lda_cohen = pd.read_csv(folder + 'lda_cohen_cuartil.csv')
med_cohen = pd.read_csv(folder + 'med_cohen_cuartil.csv')
lda_rho = pd.read_csv(folder + 'lda_spearman_cuartil.csv')
med_rho = pd.read_csv(folder + 'med_spearman_cuartil.csv')

pathfig = '/home/usuario/Escritorio/Eric/figuras/clustering/primer_cuartil/'
x = range(-100,1000,4)

for n in range(2): #por cada uno de los dos primeros CP
    
    fig, ax = plt.subplots()
    fig.set_size_inches(15,10.5)
    ax.set_facecolor("white") 
        
    # set the x-spine (see below for more info on `set_position`)
    ax.spines['left'].set_position(('axes',0))
    ax.spines['left'].set_color('black')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()

    # set the y-spine
    ax.spines['bottom'].set_position(('axes',0))
    ax.spines['bottom'].set_color('black')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()
    ax.margins(x=0)
    
    plt.tick_params(axis='both', labelsize=27)
    # plt.locator_params(axis='both', nbins=5)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    plt.xlabel('Time (ms)', fontsize=33, labelpad=30)
    plt.ylabel("Cohen's d",
               fontsize=33, labelpad=30)
    y = lda_cohen[str(n)]
    y2 = med_cohen[str(n)]
    plt.plot(x,y)
    plt.plot(x,y2)
    plt.savefig(pathfig + 'cohen_cp' + str(n+1))
    plt.show() 

#%% VARIANZA DE CADA CATEGORIA EN CADA CP
std_scale = StandardScaler()
std_scale.fit(X2)
X_scaled_0 = std_scale.transform(X2)
pca = PCA(n_components=4)
pca.fit(X_scaled_0)

X_pca_0 = pca.transform(X_scaled_0)
X_pca_0 = X_pca_0.T
dfc = pd.read_csv('C:/Users/xochipilli/Documents/things-eeg/junio/prueba_grande/nuevos/cohen_rho/categorias/categorias_conceptos.csv')
dfc12 = pd.concat([dfc]*12).sort_index()
dfc12 = dfc12.reset_index()
dfc12.drop(columns=['index'], inplace=True)
dfc12 = pd.concat([dfc12, pd.DataFrame(X_pca_0.T)],axis=1)
dfc12.columns = ['concepto', 'categoria', '0', '1', '2', '3']        

list_otra = [ast.literal_eval(i) for i in dfc['otra'].to_list()]
categorias = list(set([i for o in list_otra for i in o]))
dfc12['categoria'] = [ast.literal_eval(i) for i in dfc12['categoria'].to_list()]

dic_var = {}
for c in categorias:
    dic_var[c] = {}
    for n in [0,1,2,3]:
        ind = [i  for i in range(22248) if c in dfc12['categoria'][i]]
        a = dfc12.iloc[ind][str(n)]
        dic_var[c][n] = np.var(a)
        
df_var = pd.DataFrame(dic_var)
df_var = df_var.round(2)
df_var.to_csv('C:/Users/xochipilli/Documents/things-eeg/scripts/clustering/varianza_categoria_PCA.csv', index=False)