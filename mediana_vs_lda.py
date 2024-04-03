import mne
import pandas as pd
import numpy as np
import mat73
import matplotlib.pyplot as plt
import pickle
from scipy.stats import sem

#%% FUNCIONES
def cargar_raw(sujeto):
    '''input: sujeto
    levanta el eeg de ese sujeto
    output: raw, anotaciones de los eventos'''
    
    if sujeto < 10:
        sujeto = '0' + str(sujeto)
    
    path = 'C:/Users/xochipilli/Documents/things-eeg/eegs_crudos/derivatives/eeglab/sub-{}_task-rsvp_continuous.set'.format(sujeto)   
    raw = mne.io.read_raw_eeglab(path, eog=(), preload=True, uint16_codec=None, verbose=False)
    canales = ['PO4']
    raw = raw.pick_channels(canales) #'PO8', POZ
    if len(canales)>1: 
        raw = raw.set_eeg_reference(ref_channels='average')
    raw = raw.filter(l_freq=4, h_freq=12.5)  #0.1 100
    
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    todos = [i[0] for i in events if i[2]==1]
    eventos = todos[:22248] #todas las imagenes de la prueba grande
    path_df = 'C:/Users/xochipilli/Documents/things-eeg/eegs_crudos/sub-{}/eeg/'.format(sujeto)
    archivo = path_df+'sub-{}_task-rsvp_events.csv'.format(sujeto)
    df = pd.read_csv(archivo)
    test = df.loc[df['blocksequencenumber'] !=-1]
    test['muestra'] = eventos
    
    return [raw, test]

def lista_obj(df, raw, obj, muestra):
    '''input: df, raw, obj y la muestra
    output: lista de los valores de PO4 para ese objeto en onset+muestra'''
    
    t1 = df.loc[df.objectnumber == obj,'muestra'].tolist()
    l1 = np.zeros(12)  
    for i in range(len(t1)):
        start1 = t1[i]+muestra
        sel1 = raw[0, start1:start1+1][0]
        l1[i] = float(sel1) 
       
    return l1

def mediana(l1, l2):
    '''input: lista1 y lista2
    output: proporción más grande de elementos de cada lista a cada lado
    de la mediana'''
    
    todas = np.concatenate((l1,l2))
    umbral = np.median(todas)
    inter = sum([1 for i in l1 if i < umbral])/12  
    
    #si inter es menor a 0.5, res es la proporción complementaria
    res = inter if inter>0.5 else 1-inter
    
    return res

def clasificador_lda():
    '''genera la matriz RDM para todos los sujetos para todas las muestras'''
    
    # path = 'C:/Users/xochipilli/Documents/things-eeg/eegs_crudos/derivatives/RDM/stats_RDM_full.mat'
    path = '/home/usuario/Escritorio/Eric/stats_RDM_full.mat'
    data_dict = mat73.loadmat(path)
    rdm = data_dict['mean_RDM']
    todas = [rdm[i][np.triu_indices(1854,k=1)] for i in range(275)]
    
    return todas

def lda_individual(s):
    path = 'C:/Users/xochipilli/Documents/things-eeg/data/derivatives/RDM/'
    if s<10:
        s = '0' + str(s)
    data_dict = mat73.loadmat(path + 'sub-{}_RDM_full.mat'.format(s))
    rdm = data_dict['RDM']
    todas = [np.mean(rdm[i][np.triu_indices(1854,k=1)]) for i in range(275)]
    return todas

def matriz_mediana(sujeto):
    '''input: número de sujeto (1 al 50, descartar el 6)
    output: matriz de mediana para todas las muestras, todas las comparaciones'''
    
    raw, df = cargar_raw(sujeto)
    
    matriz = np.repeat(0, 1717731*275) #guardará los promedios de las medianas de cada muestra
    matriz = matriz.reshape(1717731, 275)
    muestras = list(range(-25,250))    #muestras
    
    for m in muestras:
        k = 0
        dic = {obj: lista_obj(df, raw, obj, m) for obj in range(1854)}
        for i in range(1853):
            for j in range(i+1, 1854):
                matriz[(k,m+25)] = mediana(dic[i],dic[j])
                k += 1
    
    return matriz

def mediana_prom(sujeto):
    '''input: número de sujeto (1 al 50, descartar el 6)
    output: matriz de mediana para todas las muestras, todas las comparaciones'''
    
    raw, df = cargar_raw(sujeto)
    prom = np.zeros(275)
    
    for m in range(275):
        dic = {obj: lista_obj(df, raw, obj, m) for obj in range(1854)}
        for i in range(1853):
            for j in range(i+1, 1854):
                prom[m] += mediana(dic[i],dic[j])
    
    prom = prom/1717731
    
    return prom

def matriz_lda_promedio():
    '''devuelve la matriz promedio para todos los sujetos, 
    todas las comparaciones, todas las muestras'''
    
    path = 'C:/Users/xochipilli/Documents/things-eeg/eegs_crudos/derivatives/RDM/stats_RDM_full.mat'
    data_dict = mat73.loadmat(path)
    rdm = data_dict['mean_RDM']
    todas = []
    for i in range(275):
        todas.append(rdm[i][np.triu_indices(1854,k=1)])
        
    return todas

def error_estandar(dic, niveles):
    
    if niveles == 1:
        medias = np.zeros(275)
        for s in dic.keys():
            medias += dic[s]
        medias /= len(dic)
            
        todos = [[dic[s][i] for s in dic.keys()] for i in range(275)]
        error = [sem(t) for t in todos]
    
        # medias = np.around(medias,3)
        # error = np.around(error,3)
        
    elif niveles == 2:
        medias = {f: np.zeros(275) for f in dic[2].keys()}
        error = {f: np.zeros(275) for f in dic[2].keys()}
        
        for f in dic[2].keys(): #para cada banda de frecuencias
            media = 0
            for s in dic.keys(): #para cada sujeto
                media += dic[s][f]
            medias[f] = media/len(dic)
            
            todos = [[dic[k][f][i] for k in dic.keys()] for i in range(275)]
            error[f] = [sem(t) for t in todos]
            
    return medias,error

#%% MEDIANA PARA TODOS LOS SUJETOS VALIDOS
#path = 'C:/Users/xochipilli/Documents/things-eeg/scripts/mediana/'
#path = 'C:/Users/exper/Todo/Facultad/Doc/Cocuco/scripts/mediana/'
path = '/home/usuario/Escritorio/Eric/scripts/mediana/'
sujetos = set(range(1,51)) - {6}
        
# dic_mediana = {}
# for s in sujetos:
#     print(s)
#     dic_mediana[s] = mediana_prom(s)
#     with open(path + 'dic_mediana.pkl', 'wb') as f:
#           pickle.dump(dic_mediana, f)

# dic_mediana['media'], dic_mediana['error'] = error_estandar(dic_mediana, 1)
# with open(path + 'dic_mediana.pkl', 'wb') as f:
#       pickle.dump(dic_mediana, f)    
#%% LDA PARA TODOS LOS SUJETOS

# dic_lda = {}
# for s in sujetos:
#     dic_lda[s] = lda_individual(s)
    # with open(path + 'dic_lda.pkl', 'wb') as f:
    #       pickle.dump(dic_lda, f)

# dic_lda['media'], dic_lda['error'] = error_estandar(dic_lda, 1)
# with open(path + 'dic_lda.pkl', 'wb') as f:
#       pickle.dump(dic_lda, f)    
      
#%% COMPARO LOS VALORES PROMEDIOS CON LOS VALORES DE AZAR DE CADA MÉTODO

with open(path + 'dic_lda.pkl', 'rb') as f:
    lda = pickle.load(f)  

with open(path + 'dic_mediana.pkl', 'rb') as f:
    med = pickle.load(f)  
        
# media = [np.mean(clasificador_lda()[i]) for i in range(275)]

#%%
fig, ax = plt.subplots()
ax.set_facecolor("white")
fig.set_size_inches(15, 10.5)

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

media = lda['media']

y1 = (np.array(media) - 0.5)*2
y1menos = ((np.array(media) - lda['error']) - 0.5)*2
y1mas = ((np.array(media) + lda['error']) - 0.5)*2

x = range(-100,1000,4)
plt.plot(x, y1, label ='Original LDA')
plt.fill_between(x, y1menos, y1mas, alpha=0.2)
# plt.fill_between(x, lda['media'] - lda['error'], lda['media'] + lda['error'],
#                   alpha=0.2)

y2 = (np.array(med['media']) - 0.5788)/0.4212
y2menos = (np.array(med['media'] - med['error'])  - 0.5788)/0.4212
y2mas = (np.array(med['media'] + med['error'])  - 0.5788)/0.4212

plt.plot(x,y2, label='Median')
plt.fill_between(x, y2menos, y2mas, alpha=0.2)

plt.xlabel('Time (ms)', fontsize=33, labelpad=30)
plt.tick_params(axis='both', labelsize=27)
plt.ylabel("Accuracy over randomness",  fontsize=33, labelpad=10)
plt.legend(loc='best', fontsize=25)
plt.plot(x, [0]*275, color='k')
plt.show()
