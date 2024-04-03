import mne
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy
from sklearn.decomposition import PCA
import random
import pickle
from scipy.stats import sem

#%% FUNCIONES
def voltajes(sujeto):
    '''input: sujeto
    levanta el eeg de ese sujeto
    output: raw, anotaciones de los eventos'''
    
    if sujeto < 10:
        sujeto = '0' + str(sujeto)
    
    path = 'C:/Users/xochipilli/Documents/things-eeg/eegs_crudos/derivatives/eeglab/sub-{}_task-rsvp_continuous.set'.format(sujeto)   
    raw = mne.io.read_raw_eeglab(path, eog=(), preload=True, uint16_codec=None, verbose=False)
    canales = ['PO4']
    raw = raw.pick_channels(canales) 
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
    
    dic = {i:0 for i in test['stimname']}
    for imagen in test['stimname']:
        t1 = test.loc[test.stimname == imagen,'muestra']
        start = int(t1)
        dic[imagen] = np.array(raw[0, start-25:start+250][0])

    return dic

def mejorar_modelo(X_pca):
    restantes = [p for p in pendientes]
    suj = random.sample(list(sujetos), 33)
    resto = sujetos - set(suj)

    v_suj = {s: {m:[sujeto_voltajes[s][n][m] for n in nombres] for m in range(25,105,2)} for s in suj}
    v_resto = {r: {m:[sujeto_voltajes[r][n][m] for n in nombres] for m in range(25,105,2)} for r in resto}

    mejor = puntaje(v_suj, X_pca)
    ref_resto = puntaje(v_resto, X_pca)
    
    variables = []
    puntajes_suj, puntajes_resto = [round(mejor,3)], [round(ref_resto,3)]
    while len(restantes)>7:
        mejor = 0
        for p in restantes:
            saco = variables + [p]
            rho = saco_pruebo(X, saco, v_suj)
            if rho > mejor:
                mejor = rho
                var_mejor = p
        restantes.remove(var_mejor)
        puntajes_suj.append(round(mejor,3))
        variables.append(var_mejor) 
        
        X3 = X.copy()     
        X3.drop(columns=variables, inplace=True)
        std_scale = StandardScaler()
        std_scale.fit(X3)
        X_scaled = std_scale.transform(X3)
        pca = PCA(n_components=1)
        pca.fit(X_scaled)
        X_pca = pca.transform(X_scaled).T
        
        puntajes_resto.append(round(puntaje(v_resto, X_pca),3))
        
    return variables, puntajes_suj, puntajes_resto

def saco_pruebo(X, saco, dic):
    X3 = X.copy()
    X3.drop(columns=saco, inplace=True)
    std_scale = StandardScaler()
    std_scale.fit(X3)
    X_scaled = std_scale.transform(X3)
    pca = PCA(n_components=1)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled).T
    rho = puntaje(dic, X_pca)
    return rho

def puntaje(dic, X_pca):
    media = np.zeros(40)
    for s in dic.keys():
        media += [scipy.stats.spearmanr(dic[s][m], X_pca[0])[0] for m in range(25,105,2)]
    
    res = sum(abs(media/len(dic)))
    return res

def error_estandar(dic, niveles):
    
    if niveles == 1:
        medias = np.zeros(275)
        for s in dic.keys():
            medias += dic[s]
        medias /= len(dic)
            
        todos = [[dic[s][i] for s in dic.keys()] for i in range(275)]
        error = [sem(t) for t in todos]
            
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

#%% VOLTAJES POR IMAGEN
# df = pd.read_csv('C:/Users/xochipilli/Documents/things-eeg/dataframe.csv')
df = pd.read_csv('/home/usuario/Escritorio/Eric/dataframe.csv')
nombres = df['stimname'].to_list()
nombres.sort()

sujetos = set(range(1,51))
# sujeto_voltajes = {s: voltajes(s) for s in sujetos}

# path = 'C:/Users/xochipilli/Documents/things-eeg/scripts/pca/'
path = '/home/usuario/Escritorio/Eric/scripts/pca/'

# with open(path + 'sujeto_voltajes.pkl', 'wb') as f:
#     pickle.dump(sujeto_voltajes, f)
    
with open(path + 'sujeto_voltajes.pkl', 'rb') as f:
    sujeto_voltajes = pickle.load(f)    
        
#%% MATRIZ DE LUZ
# X = pd.read_csv('C:/Users/xochipilli/Documents/things-eeg/junio/luminancia.csv')
X = pd.read_csv(path + 'luminancia.csv')

saco = ['concepto', 'item']
X.drop(columns=saco, inplace=True)

pendientes = ['oriEn_1', 'oriEn_2', 'oriEn_3', 'oriEn_4', 'oriEn_5', 'oriEn_6',
       'oriEn_7', 'oriEn_8', 'oriEn_9', 'oriEn_10', 'oriEn_11', 'oriEn_12',
       'oriEn_13', 'oriEn_14', 'oriEn_15', 'oriEn_16', 'oriEn_17', 'sfEn_1',
       'sfEn_2', 'sfEn_3', 'sfEn_4', 'sfEn_5', 'sfEn_6', 'sfEn_7', 'sfEn_8',
       'RMS', 'lum', 'r', 'g', 'b']

X2 = X.copy()
std_scale = StandardScaler()
std_scale.fit(X2)
X_scaled = std_scale.transform(X2)
pca = PCA(n_components=1)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled).T

#%% ITERO 20 VECES y BUSCO LAS MEJORES VARIABLES SIN SOBREAJUSTAR
resultados = [mejorar_modelo() for _ in range(20)]
    
'''las variables que en general aparecieron en 20 corridas son:'''

seleccion = ['RMS', 'oriEn_1', 'oriEn_10', 'oriEn_11', 'oriEn_13', 'oriEn_15', 
             'oriEn_16', 'oriEn_17', 'oriEn_2', 'oriEn_3', 'oriEn_5', 'oriEn_7', 
             'oriEn_8', 'oriEn_9', 'sfEn_2', 'sfEn_3', 'sfEn_4', 'sfEn_8',
             'oriEn_4', 'r', 'oriEn_12', 'sfEn_1']

#%%
X2.drop(columns=seleccion, inplace=True)
std_scale = StandardScaler()
std_scale.fit(X2)
X_scaled = std_scale.transform(X2)
pca = PCA(n_components=3)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled).T

#VEO LA COMPOSICIÓN DE LOS DOS PRIMEROS COMPONENTES PRINCIPALES
print('Features = {}'.format(X2.columns))
print('PCA1 = {}'.format(pca.components_[0]))
print('PCA2 = {}'.format(pca.components_[1]))

dic_radar = {'PCA1':pca.components_[0], 'PCA2':pca.components_[1]}
df_radar = pd.DataFrame.from_dict(dic_radar).T
df_radar.columns = X2.columns

# df_radar.to_csv(path + 'df_radar.csv', index=False)
sujetos = sujetos - {6}

muestras_sujeto_1 = {s:(np.zeros(275),np.zeros(275)) for s in sujetos}  #rho y p
muestras_sujeto_2 = {s:(np.zeros(275),np.zeros(275)) for s in sujetos}
muestras_sujeto_3 = {s:(np.zeros(275),np.zeros(275)) for s in sujetos}
for s in sujetos:
    rho1, rho2, rho3 = np.zeros(275), np.zeros(275), np.zeros(275)
    p1, p2, p3 = np.zeros(275), np.zeros(275), np.zeros(275)
    for m in range(275):
        rho1[m], p1[m] = scipy.stats.spearmanr([sujeto_voltajes[s][n][m] for n in nombres], X_pca[0])
        rho2[m], p2[m] = scipy.stats.spearmanr([sujeto_voltajes[s][n][m] for n in nombres], X_pca[1])
        rho3[m], p3[m] = scipy.stats.spearmanr([sujeto_voltajes[s][n][m] for n in nombres], X_pca[2])
    muestras_sujeto_1[s] = rho1, p1
    muestras_sujeto_2[s] = rho2, p2
    muestras_sujeto_3[s] = rho3, p3

for d in [muestras_sujeto_1, muestras_sujeto_2, muestras_sujeto_3]:
    d['media'], d['error'] = error_estandar(d, 1)

muestras_sujeto = {'cp1': muestras_sujeto_1, 'cp2': muestras_sujeto_2, 'cp3': muestras_sujeto_3}

# with open(path + 'spearman_sujeto.pkl', 'wb') as f:
#     pickle.dump(muestras_sujeto, f)

with open(path + 'spearman_sujeto.pkl', 'rb') as f:
    muestras_sujeto = pickle.load(f)   
