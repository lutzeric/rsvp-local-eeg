import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import sem
from matplotlib.ticker import FormatStrFormatter

#%% FUNCIONES

#path = '/home/usuario/Escritorio/Eric/xochipilli/scripts/mejor_canal/'
path = '/home/usuario/Escritorio/Eric/scripts/mejor_canal/'

combinaciones = [(i,j) for i in range(199) for j in range(i+1,200)]

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

def pre_filtros(sujeto):
    '''input: sujeto
    output: cohen para muestras 20 a 100, canal PO4, frecuencia 4-12.5 Hz'''
    
    path = f'/home/usuario/Escritorio/Eric/eegs_crudos/derivatives/eegs_ejemplo/sub-{sujeto:02d}_task-rsvp_continuous.set'
    raw = mne.io.read_raw_eeglab(path, eog=(), preload=True, uint16_codec=None, verbose=False)
    raw = raw.pick_channels(['PO4'])
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    todos = [i[0] for i in events if i[2]==1]
    eventos = todos[-2400:] #solamente las imagenes de la prueba pequeña
    path_df = f'/home/usuario/Escritorio/Eric/eegs_crudos/sub-{sujeto}/eeg/sub-{sujeto}_task-rsvp_events.csv'    
    df = pd.read_csv(path_df)
    test = df.loc[df['blocksequencenumber'] ==-1]
    test['presentacion'] = eventos
    #tiempos de presentacion de cada objeto (o:12)
    dic_p = {o: test.loc[test.teststimnumber == o,'presentacion'].tolist() for o in range(200)}
   
    return raw, dic_p

def phase(raw, dic_p, fase):
    raw = raw.filter(l_freq=4, h_freq=12.5, phase=fase)

    suma_cohen = np.zeros(275)
    for muestra in range(-25,250):
        #voltajes de cada objeto (200:12)

        dic_v = {o: [float(raw[0, dic_p[o][i]+muestra:dic_p[o][i]+muestra+1][0]) for i in range(12)] for o in range(200)}
        cohen = sum(abs(cohend(dic_v[i],dic_v[j])) for (i,j) in combinaciones)
        suma_cohen[muestra+25] = cohen/19900
    
    return suma_cohen

#%%
raw, dic_p = pre_filtros(38)
fases = ['zero', 'zero-double', 'minimum']
dic = {f:phase(raw,dic_p,f) for f in fases}
with open(path + 'filtros.pkl', 'wb') as f:
    pickle.dump(dic, f)

