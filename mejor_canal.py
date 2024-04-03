import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.stats import sem
from matplotlib.ticker import FormatStrFormatter

#%% FUNCIONES
'''esto me va a dar una lista de 275 valores de promedio de Cohen para cada sujeto, 
para cada canal. Lo que voy a hacer después es un heatmap con el Cohen promedio de los 48
sujetos a lo largo de las muestras. La región más iluminada es la de el/los mejores canales'''

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


def cohen_canal(sujeto):
    '''input: sujeto
    output: diccionario de 64 entradas, cada una con una lista de 275 cohens promedio, 
    uno por muestra, para cada canal'''
    
    if sujeto < 10:
        sujeto = '0' + str(sujeto)
    
    path = 'C:/Users/xochipilli/Documents/things-eeg/eegs_crudos/derivatives/eeglab/sub-{}_task-rsvp_continuous.set'.format(sujeto)   
    raw_c = mne.io.read_raw_eeglab(path, eog=(), preload=True, uint16_codec=None, verbose=False)
    raw_c = raw_c.filter(l_freq=0.1, h_freq=100)
    
    events, event_dict = mne.events_from_annotations(raw_c, verbose=False)
    todos = [i[0] for i in events if i[2]==1]
    eventos = todos[-2400:] #solamente las imagenes de la prueba pequeña
    path_df = 'C:/Users/xochipilli/Documents/things-eeg/eegs_crudos/sub-{}/eeg/'.format(sujeto)
    archivo = path_df+'sub-{}_task-rsvp_events.csv'.format(sujeto)
    df = pd.read_csv(archivo)
    test = df.loc[df['blocksequencenumber'] ==-1]
    test['presentacion'] = eventos
        
    #tiempos de presentacion de cada objeto (o:12)
    dic_p = {o: test.loc[test.teststimnumber == o,'presentacion'].tolist() for o in range(200)}
    suj_canal = {c:0 for c in canales}
    for canal in canales:
        raw = raw_c.copy()
        raw = raw.pick_channels([canal])
        total_cohen = np.zeros(275)
        
        for muestra in range(-25,250):
            #voltajes de cada objeto (200:12)
            dic_v = {o: [float(raw[0, dic_p[o][i]+muestra:dic_p[o][i]+muestra+1][0]) for i in range(12)] for o in range(200)}
            cohen = sum(abs(cohend(dic_v[i],dic_v[j])) for (i,j) in combinaciones)
            total_cohen[muestra+25] = cohen/19900
    
        suj_canal[canal] = total_cohen
    
    return suj_canal


def cohen_frecuencia(sujeto):
    '''input: sujeto
    output: diccionario de 64 entradas, cada una con una lista de 275 cohens promedio, 
    uno por muestra, para cada canal'''
    
    if sujeto < 10:
        sujeto = '0' + str(sujeto)
    
    path = 'C:/Users/xochipilli/Documents/things-eeg/eegs_crudos/derivatives/eeglab/sub-{}_task-rsvp_continuous.set'.format(sujeto)   
    raw_c = mne.io.read_raw_eeglab(path, eog=(), preload=True, uint16_codec=None, verbose=False)
    raw_c = raw_c.set_eeg_reference(ref_channels='average')
    
    events, event_dict = mne.events_from_annotations(raw_c, verbose=False)
    todos = [i[0] for i in events if i[2]==1]
    eventos = todos[-2400:] #solamente las imagenes de la prueba pequeña
    path_df = 'C:/Users/xochipilli/Documents/things-eeg/eegs_crudos/sub-{}/eeg/'.format(sujeto)
    archivo = path_df+'sub-{}_task-rsvp_events.csv'.format(sujeto)
    df = pd.read_csv(archivo)
    test = df.loc[df['blocksequencenumber'] ==-1]
    test['presentacion'] = eventos
    
    #tiempos de presentacion de cada objeto (o:12)
    dic_p = {o: test.loc[test.teststimnumber == o,'presentacion'].tolist() for o in range(200)}
    
    suj_banda = {b:0 for b in bandas}
    for b in bandas:
        raw = raw_c.copy()
        raw = raw.filter(l_freq=b[0], h_freq=b[1])
        total_cohen = np.zeros(275)
        for muestra in range(-25,250):
            
            #voltajes de cada objeto (200:12)
            dic_v = {o: [float(raw[c, dic_p[o][i]+muestra:dic_p[o][i]+muestra+1][0]) for c in range(64) for i in range(12)] for o in range(200)}           
            cohen = sum(abs(cohend(dic_v[i],dic_v[j])) for (i,j) in combinaciones)
            total_cohen[muestra+25] = cohen/19900
        suj_banda[b] = total_cohen
    
    return suj_banda


def cohen_general(sujeto):
    '''input: sujeto
    output: diccionario de 64 entradas, cada una con una lista de 275 cohens promedio, 
    uno por muestra, para cada canal'''
    
    if sujeto < 10:
        sujeto = '0' + str(sujeto)
    
    path = 'C:/Users/xochipilli/Documents/things-eeg/eegs_crudos/derivatives/eeglab/sub-{}_task-rsvp_continuous.set'.format(sujeto)   
    raw = mne.io.read_raw_eeglab(path, eog=(), preload=True, uint16_codec=None, verbose=False)
    # raw = raw.filter(l_freq=0.1, h_freq=100)
    raw = raw.set_eeg_reference(ref_channels='average')
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    todos = [i[0] for i in events if i[2]==1]
    eventos = todos[-2400:] #solamente las imagenes de la prueba pequeña
    path_df = 'C:/Users/xochipilli/Documents/things-eeg/eegs_crudos/sub-{}/eeg/'.format(sujeto)
    archivo = path_df+'sub-{}_task-rsvp_events.csv'.format(sujeto)
    df = pd.read_csv(archivo)
    test = df.loc[df['blocksequencenumber'] ==-1]
    test['presentacion'] = eventos
        
    #tiempos de presentacion de cada objeto (o:12)
    dic_p = {o: test.loc[test.teststimnumber == o,'presentacion'].tolist() for o in range(200)}

    total_cohen = np.zeros(275)
    
    for muestra in range(-25,250):
        #voltajes de cada objeto (200:12)
        dic_v = {o: [float(raw[c, dic_p[o][i]+muestra:dic_p[o][i]+muestra+1][0]) for c in range(64) for i in range(12)] for o in range(200)}
        cohen = sum(abs(cohend(dic_v[i],dic_v[j])) for (i,j) in combinaciones)
        total_cohen[muestra+25] = cohen/19900
    
    return total_cohen


def cohen_mejor(sujeto):
    '''input: sujeto
    output: cohen para muestras 20 a 100, canal PO4, frecuencia 4-12.5 Hz'''
    
    if sujeto < 10:
        sujeto = '0' + str(sujeto)
    
    path = 'C:/Users/xochipilli/Documents/things-eeg/eegs_crudos/derivatives/eeglab/sub-{}_task-rsvp_continuous.set'.format(sujeto)   
    raw = mne.io.read_raw_eeglab(path, eog=(), preload=True, uint16_codec=None, verbose=False)
    raw = raw.filter(l_freq=4, h_freq=12.5)
    raw = raw.pick_channels(['PO4'])
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    todos = [i[0] for i in events if i[2]==1]
    eventos = todos[-2400:] #solamente las imagenes de la prueba pequeña
    path_df = 'C:/Users/xochipilli/Documents/things-eeg/eegs_crudos/sub-{}/eeg/'.format(sujeto)
    archivo = path_df+'sub-{}_task-rsvp_events.csv'.format(sujeto)
    df = pd.read_csv(archivo)
    test = df.loc[df['blocksequencenumber'] ==-1]
    test['presentacion'] = eventos
        
    #tiempos de presentacion de cada objeto (o:12)
    dic_p = {o: test.loc[test.teststimnumber == o,'presentacion'].tolist() for o in range(200)}

    total_cohen = np.zeros(275)
    
    for muestra in range(-25,250):
        #voltajes de cada objeto (200:12)
        dic_v = {o: [float(raw[0, dic_p[o][i]+muestra:dic_p[o][i]+muestra+1][0]) for i in range(12)] for o in range(200)}
        cohen = sum(abs(cohend(dic_v[i],dic_v[j])) for (i,j) in combinaciones)
        total_cohen[muestra+25] = cohen/19900
    
    return total_cohen


def cohen_po4_4_8(sujeto):
    '''input: sujeto
    output: cohen para muestras 20 a 100, canal PO4, frecuencia 4-8 Hz'''
    
    if sujeto < 10:
        sujeto = '0' + str(sujeto)
    
    path = 'C:/Users/xochipilli/Documents/things-eeg/eegs_crudos/derivatives/eeglab/sub-{}_task-rsvp_continuous.set'.format(sujeto)   
    raw = mne.io.read_raw_eeglab(path, eog=(), preload=True, uint16_codec=None, verbose=False)
    raw = raw.filter(l_freq=4, h_freq=8)
    raw = raw.pick_channels(['PO4'])
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    todos = [i[0] for i in events if i[2]==1]
    eventos = todos[-2400:] #solamente las imagenes de la prueba pequeña
    path_df = 'C:/Users/xochipilli/Documents/things-eeg/eegs_crudos/sub-{}/eeg/'.format(sujeto)
    archivo = path_df+'sub-{}_task-rsvp_events.csv'.format(sujeto)
    df = pd.read_csv(archivo)
    test = df.loc[df['blocksequencenumber'] ==-1]
    test['presentacion'] = eventos
        
    #tiempos de presentacion de cada objeto (o:12)
    dic_p = {o: test.loc[test.teststimnumber == o,'presentacion'].tolist() for o in range(200)}

    total_cohen = np.zeros(275)
    
    for muestra in range(-25,250):
        #voltajes de cada objeto (200:12)
        dic_v = {o: [float(raw[0, dic_p[o][i]+muestra:dic_p[o][i]+muestra+1][0]) for i in range(12)] for o in range(200)}
        cohen = sum(abs(cohend(dic_v[i],dic_v[j])) for (i,j) in combinaciones)
        total_cohen[muestra+25] = cohen/19900
    
    return total_cohen


def cohen_po4_4_30(sujeto):
    '''input: sujeto
    output: cohen para muestras 20 a 100, canal PO4, frecuencia 4-30 Hz'''
    
    if sujeto < 10:
        sujeto = '0' + str(sujeto)
    
    path = 'C:/Users/xochipilli/Documents/things-eeg/eegs_crudos/derivatives/eeglab/sub-{}_task-rsvp_continuous.set'.format(sujeto)   
    raw = mne.io.read_raw_eeglab(path, eog=(), preload=True, uint16_codec=None, verbose=False)
    raw = raw.filter(l_freq=4, h_freq=30)
    raw = raw.pick_channels(['PO4'])
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    todos = [i[0] for i in events if i[2]==1]
    eventos = todos[-2400:] #solamente las imagenes de la prueba pequeña
    path_df = 'C:/Users/xochipilli/Documents/things-eeg/eegs_crudos/sub-{}/eeg/'.format(sujeto)
    archivo = path_df+'sub-{}_task-rsvp_events.csv'.format(sujeto)
    df = pd.read_csv(archivo)
    test = df.loc[df['blocksequencenumber'] ==-1]
    test['presentacion'] = eventos
        
    #tiempos de presentacion de cada objeto (o:12)
    dic_p = {o: test.loc[test.teststimnumber == o,'presentacion'].tolist() for o in range(200)}

    total_cohen = np.zeros(275)
    
    for muestra in range(-25,250):
        #voltajes de cada objeto (200:12)
        dic_v = {o: [float(raw[0, dic_p[o][i]+muestra:dic_p[o][i]+muestra+1][0]) for i in range(12)] for o in range(200)}
        cohen = sum(abs(cohend(dic_v[i],dic_v[j])) for (i,j) in combinaciones)
        total_cohen[muestra+25] = cohen/19900
    
    return total_cohen


def error_estandar(dic, niveles):
    
    if niveles == 1:
        medias = np.zeros(275)
        for s in sujetos:
            medias += dic[s]
        medias /= len(dic)
            
        todos = [[dic[s][i] for s in sujetos] for i in range(275)]
        error = [sem(t) for t in todos]
            
    elif niveles == 2:
        medias = {f: np.zeros(275) for f in dic[2].keys()}
        error = {f: np.zeros(275) for f in dic[2].keys()}
        
        for f in dic[2].keys(): #para cada banda de frecuencias
            media = 0
            for s in sujetos: #para cada sujeto
                media += dic[s][f]
            medias[f] = media/len(dic)
            
            todos = [[dic[k][f][i] for k in sujetos] for i in range(275)]
            error[f] = [sem(t) for t in todos]
            
    return medias,error

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

#%%
#path = '/home/usuario/Escritorio/Eric/xochipilli/scripts/mejor_canal/'
# path = 'C:/Users/exper/Todo/Facultad/Doc/Cocuco/scripts/mejor_canal/'
path = '/home/usuario/Escritorio/Eric/scripts/mejor_canal/'

combinaciones = [(i,j) for i in range(199) for j in range(i+1,200)]

canales = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5',
           'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6',
           'CP2', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7',
           'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 
           'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 
           'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2', 'FCz', 'Cz']

bandas = [(0.5,4), (4,8), (8,12.5), (12.5,30), (30,50), (50,70), (70,100)]
sujetos = set(range(2,51))
sujetos -= {6}

#%% COHEN GENERAL: TODOS LOS CANALES EN BANDA ANCHA (CRUDO)
# dic_cohen_general = {s:cohen_general(s) for s in sujetos}
# with open(path + 'dic_cohen_general.pkl', 'wb') as f:
#       pickle.dump(dic_cohen_general, f)
      
#%% BARRIDO PARA CANALES POR SEPARADO
# dic_suj_canal = {s:cohen_canal(s) for s in sujetos}
# with open(path + 'dic_cohen_canal.pkl', 'wb') as f:
#     pickle.dump(dic_cohen_canal, f)

#%% CARGO LOS DICCIONARIOS E INFORMACION UTIL 
total = pd.DataFrame()
with open(path + 'dic_cohen_canales.pkl', 'rb') as f:
    dic_cohen_canales = pickle.load(f)    
for s in sujetos:
    df = pd.DataFrame(dic_cohen_canales[s])
    total = total.add(df, fill_value=0)
total = total.divide(len(sujetos))
total.index = range(-100,1000,4)

buenos = ['P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'POz', 'PO4', 'PO8', 'P2']
x = range(-100,1000,4)

#%%
#HEATMAP DE CANALES cmap = RdBu_r

sns.set(rc={'figure.figsize':(25,11)})

paleta = 'coolwarm'#'YlGnBu'
ax = sns.heatmap(total.T, cmap=paleta, xticklabels=False, yticklabels=False,
                 cbar_kws={"aspect": 10, "pad": 0.03})

plt.tick_params(axis='both', labelsize=27)
plt.xlabel('Time (ms)', fontsize=30, labelpad=15)
plt.ylabel('Channel', fontsize=30, labelpad=20)

# use matplotlib.colorbar.Colorbar object
cbar = ax.collections[0].colorbar

cbar.ax.tick_params(labelsize=25)
cbar.set_label("Cohen's d", size=30,labelpad=30)

plt.show()

#%%
#PROMEDIO EN FUNCION DEL TIEMPO DE LA d de COHEN PARA TODOS LOS CANALES

sns.set(rc={'figure.figsize':(25,11)})
ax = plt.axes()
ax.set_facecolor("white")

# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.tick_params(axis='both', labelsize=27)
plt.xlabel('Time', fontsize=33, labelpad=30)
plt.ylabel("Cohen's d", fontsize=33, labelpad=30)

y = total.T.mean()
plt.plot(x,y, linewidth=3.5, color='b')

plt.yticks(np.arange(0.33, 0.351, step=0.01))

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

plt.show()


#%%
# COHEN EN FUNCION DEL TIEMPO PARA LOS MEJORES CANALES

sns.reset_orig()
fig, ax = plt.subplots()
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

fig.set_size_inches(15, 10.5)

x = range(-100,1000, 4)

media = dic_cohen_canales['media']
error = dic_cohen_canales['error']

for b in buenos:
    ax.plot(x, media[b], label=b)
    ax.fill_between(x, media[b] - error[b], media[b] + error[b],
                       alpha=0.1)
    
plt.tick_params(axis='both', labelsize=27)
plt.legend(loc='best', fontsize=25)
plt.xlabel('Time (ms)', fontsize=33, labelpad=30)
plt.ylabel("Cohen's d ", fontsize=33, labelpad=30)
plt.text(146,0.425,'PO4', fontsize=30)

plt.show()

#%% BARRIDO PARA FRECUENCIAS
# dic_cohen_frec = {s: 0 for s in sujetos}
# with open(path + 'dic_cohen_frec.pkl', 'wb') as f:
#       pickle.dump(dic_cohen_frec, f)
          
#%% GRAFICO FRECUENCIAS (PUEDE QUE NO LO USE)
with open(path + 'dic_cohen_frec.pkl', 'rb') as f:
    dic_cohen_frec= pickle.load(f)    

x = range(-100,1000, 4)

media, error = error_estandar(dic_cohen_frec, 2) #, [13,18,25])
bandas = ['0.5-4', '4-8', '8-12.5', '12.5-30', '30-50', '50-70', '70-100']

for b, k in enumerate(media.keys()):
    plt.plot(x, media[k], label=bandas[b])
    # plt.fill_between(x, media[k] - error[k], media[k] + error[k],
    #                   color='black', alpha=0.2)

#cruda
# with open(path + 'dic_cohen_canales.pkl', 'rb') as f:
#     dic_cohen_canal= pickle.load(f)    
# media, error = error_estandar(dic_cohen_general, 1)
# plt.plot(x, media, label='Banda ancha')
# plt.fill_between(x, media - error, media + error,
#                   color='black', alpha=0.2)

plt.legend(loc='best')
plt.xlabel("Time (ms)")
plt.ylabel("Cohen's D")
plt.ylim(-0.1,0.1)
plt.show()

# sns.set(rc={'figure.figsize':(11.7,8.27)})
# sns.heatmap(total_frec, xticklabels=True, cmap="viridis")

#%% COHEN PARA PO4 y 4-12.5 Hz
# dic_cohen_mejor = {s:cohen_mejor(s) for s in sujetos}
# with open(path + 'dic_cohen_mejor.pkl', 'wb') as f:
#       pickle.dump(dic_cohen_mejor, f)
      
#%% PO4 de 4-8 Hz
# po4_4_8 = {s:np.zeros(275) for s in sujetos}
# for s in sujetos:
#     po4_4_8[s] = cohen_po4_4_8(s)
#     with open(path + 'dic_cohen_po4_4_8.pkl', 'wb') as f:
#         pickle.dump(po4_4_8, f)

#%% PO4 de 4-30 Hz
# po4_4_30 = {s:np.zeros(275) for s in sujetos}
# for s in sujetos:
#     po4_4_30[s] = cohen_po4_4_30(s)
#     with open(path + 'dic_cohen_po4_4_30.pkl', 'wb') as f:
#           pickle.dump(po4_4_30, f)
          
#%% 
# solo PO4, banda ancha
with open(path + 'dic_cohen_po4.pkl', 'rb') as f:
    dic_cohen_po4 = pickle.load(f)    

# PO4 4-8 Hz
with open(path + 'dic_cohen_po4_4_8.pkl', 'rb') as f:
    dic_po4_4_8 = pickle.load(f)   
    
# PO4 4-30 Hz
with open(path + 'dic_cohen_po4_4_30.pkl', 'rb') as f:
    dic_po4_4_30 = pickle.load(f)   
        
# mejor
with open(path + 'dic_cohen_mejor.pkl', 'rb') as f:
    dic_cohen_mejor= pickle.load(f)   
        
# solo ciertas bandas de frecuencias, todos los canales
# with open(path + 'dic_cohen_frec.pkl', 'rb') as f:
#     dic_cohen_frec= pickle.load(f)  
# media, error = error_estandar(dic_cohen_frec, 2)

# k = (4, 8)
# plt.plot(x, media[k], label='4-8')
# plt.fill_between(x, media[k] - error[k], media[k] + error[k],
#                   color='black', alpha=0.2)

# k = (8, 12.5)
# plt.plot(x, media[k], label='8-12.5')
# plt.fill_between(x, media[k] - error[k], media[k] + error[k],
#                   color='black', alpha=0.2)

#%% GRAFICOS: COMPARACION DE MODELOS

sns.reset_orig()
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

x = range(-100,1000, 4)
    
media, error = error_estandar(dic_cohen_po4, 1)
plt.plot(x, media, label='PO4 broadband', color='olive')
plt.fill_between(x, media - error, media + error,
                  color='olive', alpha=0.2)

media, error = error_estandar(dic_cohen_mejor, 1)
plt.plot(x, media, label='PO4 4-12.5 Hz', color='purple')
plt.fill_between(x, media - error, media + error,
                  color='purple', alpha=0.2)

media = dic_po4_4_8['medias']
error = dic_po4_4_8['error']
plt.plot(x, media, label='PO4 4-8 Hz', color='cyan')
plt.fill_between(x, media - error, media + error,
                  color='cyan', alpha=0.2)

media = dic_po4_4_30['medias']
error = dic_po4_4_30['error']
plt.plot(x, media, label='PO4 4-30 Hz', color='orange')
plt.fill_between(x, media - error, media + error,
                  color='orange', alpha=0.2)

# plt.plot(x, ldam, label='LDA')
# plt.plot(x, lda2m, label='LDA mejorado')
plt.tick_params(axis='both', labelsize=27)
plt.xlabel('Time (ms)', fontsize=33, labelpad=30)
plt.ylabel("Cohen's d", fontsize=33, labelpad=30)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.legend(loc='best', fontsize=25)
plt.show()