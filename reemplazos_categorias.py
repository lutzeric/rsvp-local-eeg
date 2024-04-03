import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.spatial import distance
import copy
import pickle
# from knockknock import telegram_sender
import seaborn as sns
import scipy
from matplotlib.ticker import FormatStrFormatter

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
   
def diccionario_indices():
    con_ind = {c:[] for c in conceptos}
    ind_con = {}
    k = 0
    for i in range(1853):
        for j in range(i+1, 1854):
            con_ind[conceptos[i]].append(k)
            con_ind[conceptos[j]].append(k)
            ind_con[k] = (conceptos[i], conceptos[j])
            k += 1
    
    return con_ind, ind_con


def coincidencias_iniciales(v):
    todos_indices = np.array([con_ind[c] for c in cat_con[v]]).flatten()
    a = set([i for i in todos_indices if np.count_nonzero(todos_indices == i) == 2])
    b = set(range(1717731)) - a

    return list(a), list(b)

def mas_cercano(objetivo, misma):
    comp_cat = [cat_con[s] for s in con_cat[objetivo]]
    comp_cat = [c for i in comp_cat for c in i]
    comp_cat.extend(misma)
    
    validos = [c for c in conceptos if c not in comp_cat]
        
    dist = [abs(distance.euclidean(X.loc[objetivo], X.loc[v])) for v in validos]
    
    try:
        cercano = validos[np.argmin(dist)]
    except:
        cercano = -1

    return cercano

def busqueda_reemplazo(misma, indice):
    '''recibe una lista aleatorizada de la categoría en cuestión (misma)
    y el índice del elemento que tiene que intentar cambiar.
    Si es exitoso, devuelve la lista con el cambio hecho. Si no, 
    devuelve la misma lista que antes'''
             
    #reemplazo en la lista de conceptos de esa categoria
    reemplazo = mas_cercano(misma[indice], misma)
    dupla = (-1, -1)
        
    if reemplazo != -1:

        #devuelvo los conceptos a intercambiar
        dupla = (misma[indice], reemplazo)
        
        #elimino el objetivo de 'misma'
        misma.pop(indice)
        
    return dupla
          
#coincidencias_iniciales = cat_con(v)
def coincidencias_actuales(a, b, dupla, misma): 
    if dupla != (-1, -1):
        
        ind_obj = con_ind[dupla[0]] #lista de indices del objetivo
        ind_rem = con_ind[dupla[1]] #lista de indices del reemplazo
        
        c = set(a) - set(ind_obj) #saco los indices del objetivo
        a = list(c)
        rem_misma = [i for i in ind_rem if any([1 for r in ind_con[i] if r in misma])]
        a.extend(rem_misma) #agrego a 'a' los índices compartidos entre alguno de 'misma' y 'reemplazo'
        
        d = set(b) - set(ind_obj)
        d -= set(rem_misma)
        b = list(d)
        
        #agrego el reemplazo a 'misma'
        misma.append(dupla[1])
    
    else:
        pass
    
    return a,b

def mezclar_categoria(v, a, b):
    misma = copy.deepcopy(cat_con[v])
    random.shuffle(misma)
    lda_cohen, med_cohen = np.zeros(len(misma)), np.zeros(len(misma))
    for k in range(len(misma)-1,-1,-1):    #del final al principio de 'misma'
        dupla = busqueda_reemplazo(misma, k)
        coin, dis = coincidencias_actuales(a,b, dupla, misma)
        lda_cohen[k] = np.mean([cohend(lda[i][dis], lda[i][coin]) for i in str_i])  #disidencias - coincidencias
        med_cohen[k] = np.mean([cohend(mediana[i][dis], mediana[i][coin]) for i in str_i])
            
    return lda_cohen, med_cohen

# def promediar(dic, key):
#     prom = np.zeros(len(dic[key][0]))
#     for i in range(5):
#         prom += dic[key][i]
#     x = range(len(dic[key][0]))
#     plt.plot(x,prom)
#     plt.title(key)
#     plt.xlabel=('Iteraciones')
#     plt.ylabel=('D de Cohen')
#     plt.show()
    
#%% DATASETS. voy a usar solo Cohen de 45 a 80

# lda = cargar_df('C:/Users/xochipilli/Documents/things-eeg/junio/prueba_grande/lda_df.csv')
# mediana = cargar_df('C:/Users/xochipilli/Documents/things-eeg/junio/prueba_grande/mediana_df.csv')
lda = pd.read_csv('/home/usuario/Escritorio/Eric/prueba_grande/lda_df.csv')
mediana = pd.read_csv('/home/usuario/Escritorio/Eric/prueba_grande/mediana_df.csv')

# path = 'C:/Users/xochipilli/Documents/things-eeg/junio/prueba_grande/nuevos/cohen_rho/categorias/'
f_dic = '/home/usuario/Escritorio/Eric/scripts/categorias/'

with open(f_dic + 'cat_lda_cohen.pkl', 'rb') as f:
      cohen_lda = pickle.load(f)
with open(f_dic + 'cat_med_cohen.pkl', 'rb') as f:
      cohen_med = pickle.load(f)

with open(f_dic + 'cat_con.pkl', 'rb') as f:
    cat_con = pickle.load(f)
with open(f_dic + 'con_cat.pkl', 'rb') as f:
    con_cat = pickle.load(f)
conceptos = list(con_cat.keys())
conceptos.sort()

cohen_lda_d = pd.DataFrame.from_dict(cohen_lda)
cohen_med_d = pd.DataFrame.from_dict(cohen_med)

#solo voy a usar las muestras 45 a 80
ref_cohen_lda = cohen_lda_d.iloc[45:80].mean()
ref_cohen_med = cohen_med_d.iloc[45:80].mean()

X = pd.read_csv('/home/usuario/Escritorio/Eric/scripts/luminancia.csv')

saco = ['RMS', 'oriEn_1', 'oriEn_10', 'oriEn_11', 'oriEn_13', 'oriEn_15', 
        'oriEn_16', 'oriEn_17', 'oriEn_2', 'oriEn_3', 'oriEn_5', 'oriEn_7', 
        'oriEn_8', 'oriEn_9', 'sfEn_2', 'sfEn_3', 'sfEn_4', 'sfEn_8',
        'oriEn_4', 'r', 'oriEn_12', 'sfEn_1']

X.drop(columns=saco, inplace=True)
X = X.groupby(['concepto']).mean()
X.index= conceptos

#%% COHEN Y D PARA MATRICES DE CATEGORIAS

num = 3
str_i = [str(i) for i in range(45,80)]
dic_med, dic_lda = {}, {}

# path = 'C:/Users/xochipilli/Documents/things-eeg/scripts/reemplazos/'
path = '/home/usuario/Escritorio/Eric/scripts/'
    
con_ind, ind_con = diccionario_indices()   
# validas = ['hardware','human','body','animal','tool','natural']
validas = ['hardware', 'tool', 'natural']

#%% CORRIDAS
# print('corrida número ' + str(num))
# for v in validas:
#     dic_med[v], dic_lda[v] = [], []
#     for h in range(5):
#         try:
#             a, b = coincidencias_iniciales(v)
#             ll, mm = mezclar_categoria(v,a,b)
#             dic_lda[v].append(ll)
#             dic_med[v].append(mm)
            
            # with open(path + str(num) + 'dic_lda.pkl', 'wb') as f:
            #       pickle.dump(dic_lda, f)
            # with open(path + str(num) + 'dic_med.pkl', 'wb') as f:
            #       pickle.dump(dic_med, f)
                  
#             print(h)
    
#         except:
#             print('ERROR', v)

# with open(path + 'iteraciones/dic_lda.pkl', 'rb') as f:
#     gran_lda = pickle.load(f)
    
# with open(path + 'iteraciones/dic_med.pkl', 'rb') as f:
#     gran_med = pickle.load(f)
    
# for v in validas:
#     gran_lda[v] = []
#     gran_med[v] = []

#     for num in range(4):
#         with open(path + str(num) + 'dic_med.pkl', 'rb') as f:
#             dic_med = pickle.load(f)
#         with open(path + str(num) + 'dic_lda.pkl', 'rb') as f:
#             dic_lda = pickle.load(f)
#         gran_lda[v].extend(dic_lda[v])
#         gran_med[v].extend(dic_med[v])
    
# with open(path + 'dic_lda.pkl', 'wb') as f:
#       pickle.dump(gran_lda, f)
# with open(path + 'dic_med.pkl', 'wb') as f:
#       pickle.dump(gran_med, f)

#%% GUARDO/ABRO LOS DICCIONARIOS
# path = '/home/usuario/Escritorio/Eric/scripts/reemplazos/'

# with open(path + 'dic_med.pkl', 'rb') as f:
#     gran_med = pickle.load(f)
                          
# with open(path + 'dic_lda.pkl', 'rb') as f:
#     gran_lda = pickle.load(f)

# metodos = ['LDA ', 'Mediana ']
# sns.set(rc={'figure.figsize':(11.7,8.27)})

# #%%
# pathfig = '/home/usuario/Escritorio/Eric/figuras/categorias/reemplazos/'
# dics = [gran_lda, gran_med]
# nombre= ['lda_', 'mediana_']
# for n in range(2):
#     for key in dics[n].keys():
#         iteraciones = len(dics[n][key][0])
#         pendientes, ordenadas = [],[]
        
#         fig, ax = plt.subplots()
#         ax.set_facecolor("white") 
#         for r in range(len(dics[n][key])):
#             x = range(iteraciones)
#             y = dics[n][key][r]
#             fig.set_size_inches(18, 15)
        
#             # set the x-spine (see below for more info on `set_position`)
#             ax.spines['left'].set_position(('axes',0))
#             ax.spines['left'].set_color('black')
        
#             # turn off the right spine/ticks
#             ax.spines['right'].set_color('none')
#             ax.yaxis.tick_left()
        
#             # set the y-spine
#             ax.spines['bottom'].set_position(('axes',0))
#             ax.spines['bottom'].set_color('black')
        
#             # turn off the top spine/ticks
#             ax.spines['top'].set_color('none')
#             ax.xaxis.tick_bottom()
#             ax.margins(x=0)
            
#             plt.tick_params(axis='both', labelsize=27)
#             plt.locator_params(axis='both', nbins=5)
#             ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            
#             plt.xlabel('Iteration', fontsize=33, labelpad=30)
#             plt.ylabel('Non coincidental - coincidental',
#                        fontsize=33, labelpad=30)
#             plt.scatter(x,y, s=5, color='green')
#             m, b = np.polyfit(x,y,1)
#             pendientes.append(m)
#             ordenadas.append(b)
#             plt.plot(x, (m*x)+b, color = 'blue', alpha=0.2)
    
#         prom = np.mean(pendientes)
#         promb = np.mean(ordenadas)
#         pval = np.round(scipy.stats.ttest_1samp(pendientes, 0)[1],3)
#         plt.plot(x, prom*x+promb, color='blue', linewidth=4)
#         plt.title('Mean slope = {:.2e} p-value = {:.2e}'.format(prom,pval) + nombre[n] + key, 
#                   color='purple', fontsize=30, pad=30)
#         plt.savefig(pathfig + nombre[n] + key)
#         plt.show() 

#%%
cats[c] = 'hardware'
'''LDA'''
pendientes, ordenadas = [], []
for r in range(len(gran_lda[cats[c]])):
    y = gran_lda[cats[c]][r]
    m, b = np.polyfit(xi,y,1)
    pendientes.append(m)
    ordenadas.append(b)

pval = scipy.stats.ttest_1samp(pendientes, 0)[1]
gran_lda['estadisticos'][cats[c]] = {'pendientes': pendientes, 'ordenadas':ordenadas, 'pvalor':pval}


'''Mediana'''
pendientes, ordenadas = [], []
for r in range(len(gran_med[cats[c]])):
    y = gran_med[cats[c]][r]
    m, b = np.polyfit(xi,y,1)
    pendientes.append(m)
    ordenadas.append(b)

pval = scipy.stats.ttest_1samp(pendientes, 0)[1]
gran_med['estadisticos'][cats[c]] = {'pendientes': pendientes, 'ordenadas':ordenadas, 'pvalor':pval}
