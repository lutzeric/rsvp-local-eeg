import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as shc
from sklearn.neighbors import NearestCentroid

#%% DATASETS LUMINANCIA, LDA Y MEDIANA

# X = pd.read_csv('C:/Users/xochipilli/Documents/things-eeg/junio/luminancia.csv')
X = pd.read_csv('/home/usuario/Escritorio/Eric/scripts/pca/luminancia.csv')

# saco = ['oriEn_9', 'sfEn_8', 'RMS', 'oriEn_16', 'oriEn_10', 'oriEn_2', 'oriEn_15',
#         'sfEn_3', 'oriEn_5', 'oriEn_3', 'r', 'oriEn_11', 'oriEn_13', 'oriEn_1',
#         'oriEn_7', 'g', 'oriEn_12', 'sfEn_4', 'oriEn_4', 'sfEn_7', 'oriEn_8',
#         'sfEn_2', 'lum', 'oriEn_14', 'sfEn_5', 'item', 'oriEn_17']

saco = ['oriEn_9', 'sfEn_8', 'RMS', 'oriEn_16', 'oriEn_10', 'oriEn_2', 'oriEn_15',
        'sfEn_3', 'oriEn_5', 'oriEn_3', 'r', 'oriEn_11', 'oriEn_13', 'oriEn_1',
        'oriEn_7', 'oriEn_12', 'sfEn_4', 'oriEn_4', 'oriEn_8',
        'sfEn_2', 'oriEn_1', 'item', 'oriEn_17', 'sfEn_1']

X.drop(columns=saco, inplace=True)
X2 = X.groupby(['concepto']).mean()


# df_clusters.to_csv('/home/usuario/Escritorio/Eric/scripts/pca/df_4clusters.csv', index=None)
# df_clusters.columns =  ['OE 6', 'OE 14', 'SFE 5', 'SFE 6', 'SFE 7', 'Luminance', 'Green',
#                         'Blue']
#%% CLUSTERS JERARQUICOS
std_scale = StandardScaler()
std_scale.fit(X2)
X_scaled = std_scale.transform(X2)

# se ven 3, 4 o 5 clusters y el método del codo sugiere 4

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

'''da mejor cluster aglomerativo que K means'''
# kmeans = KMeans(n_clusters=4, random_state=0).fit(X_scaled)
# labels = kmeans.labels_
# kmeans.cluster_centers_

#da mejor con clustering aglomerativo que con Kmeans
    
#%% RADAR PLOT
clf = NearestCentroid()
clf.fit(X_scaled, etiquetas)
clf.centroids_

def radar_plot(df, title = '',):
    categories = df.columns
    N = len(categories)
    # Lo primero que hacemos es setear los ángulos a destacar en nuestro radar plot: uno para cada categoría que tengamos
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
     
    # Inicializamos una figura, teniendo en cuenta que debemos indicarle que vamos a trabajar en polares
    fig, ax = plt.subplots(figsize = (15,15), subplot_kw = {'projection': 'polar'})
     
    # Estas líneas ordenan las categorías de forma tal que la primera vaya arriba en el centro y el resto se distribuya en orden según las agujas del reloj
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
     
    # En cada uno de los ángulos, agregamos un tick con una etiqueta igual a la de la categoría en cuestión
    ax.set_xticks(angles[:-1])
    # ax.set_xticklabels(categories,
    #                    fontsize = 25)
     
    # Con estas líneas agregamos anillos que indiquen la distancia radial al centro del gráfico
    ax.set_rlabel_position(0)
    # ax.set_yticks([0.1,0.2,0.3,0.4,0.5,])
    ax.set_yticklabels([-0.5, 0, 0.5, 1, 1.5], size=20)
    #                    color = "darkgrey",
    #                    size = 10)
    # ax.set_ylim(0,11000000)

    # ¿Y mis datos, dónde están mis datos?
    # Con esta iteración, recorremos las filas del data frame, agregando cada una a la figura
    for row in df.iloc:
            values = row.values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1.5, linestyle = 'solid') # Ploteamos la línea
            # ax.fill(angles, values, 'b', alpha=0.1) # La rellenamos. Esto puede evitarse, o variar el alpha
    # Agregamos una legend que indique cuál es cada línea
    # ax.legend(loc=(-0.2,-0.3),fontsize=12)
    # Seteamos el título
    # ax.set_title(title, position=(.5, 1.2),fontsize=15,)
    
df = pd.DataFrame(dict(
    Cluster0=clf.centroids_[0], #kmeans.cluster_centers_[0],
    theta=X2.columns))

df['Cluster1'] = clf.centroids_[1] #kmeans.cluster_centers_[1] #
df['Cluster2'] = clf.centroids_[2] #kmeans.cluster_centers_[2] #
df['Cluster3'] = clf.centroids_[3] #kmeans.cluster_centers_[2] #
df = df.T
df.drop(['theta'], inplace=True)
df.columns = X2.columns
# df['lum'] = -df['lum']
df = df.astype('float')
# df2 = np.log(df)

radar_plot(df)

#%%