import mat73
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import sem

#%%
sujetos = set(range(2,51))
sujetos = sujetos - {6}
path = '/home/usuario/Escritorio/Eric/scripts/mejor_canal/'
#path = 'C:/Users/exper/Todo/Facultad/Doc/Cocuco/scripts/mejor_canal/'

#%%
def cargar_matriz_paper(sujeto):
    '''input: sujeto
    output: lista de promedios del clasificador de matlab para las 275 muestras'''
    
    # folder = 'C:/Users/xochipilli/Documents/things-eeg/data/derivatives/RDM/'
    folder = '/home/usuario/Escritorio/Eric/eegs_crudos/derivatives/4_12hz_PO4/'
    suj = f'sub-{sujeto:02d}_rdm_test_images.mat'.format(sujeto)
    path = folder + suj
    data_dict = mat73.loadmat(path)
    samples = data_dict['res']['samples']
    promedios = np.mean(samples, axis=0)

    return promedios

def cargar_matriz_matlab_mejorada(sujeto):
    '''input: sujeto
    output: lista de promedios del clasificador de matlab para las 275 muestras'''
    
    # folder = 'C:/Users/xochipilli/Documents/things-eeg/eegs_crudos/derivatives/RDM/4_12hz_PO4/'
    folder = '/home/usuario/Escritorio/Eric/eegs_crudos/derivatives/4_12hz_PO4/'
    suj = f'sub-{sujeto:02d}_po4_4_12.mat'
    path = folder + suj
    data_dict = mat73.loadmat(path)
    samples = data_dict['res']['samples']
    promedios = np.mean(samples, axis=0)
    
    return promedios

def error_estandar(dic, niveles):
    sujetos = [d for d in dic.keys() if type(d)==int]
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

#%% 
# dic_paper = {s: cargar_matriz_paper(s) for s in sujetos}
dic_mejor = {s: cargar_matriz_matlab_mejorada(s) for s in sujetos}

# path = 'C:/Users/xochipilli/Documents/things-eeg/scripts/mejor_canal/'
path = '/home/usuario/Escritorio/Eric/scripts/mejor_canal/'


# for nombre, dic in ['lda_paper.pkl', 'lda_mejor.pkl'], [dic_paper, dic_mejor]:
#     df = pd.DataFrame(dic).T
#     media = np.around(df.mean(axis=0),3).to_list()
#     error = np.around([sem(df[c].to_list()) for c in range(275)],3)
#     dic['media'] = media
#     dic['error'] = error
    with open(path + nombre, 'wb') as f:
          pickle.dump(dic, f)
          
#%% CARGO LOS DICCIONARIOS

with open(path + 'lda_paper.pkl', 'rb') as f:
    paper = pickle.load(f)  
    
with open(path + 'lda_mejorado.pkl', 'rb') as f:
    mejor = pickle.load(f) 
    
path_mediana = '/home/usuario/Escritorio/Eric/scripts/mediana/dic_mediana_chica.pkl'
with open(path_mediana, 'rb') as f:
    mediana= pickle.load(f)  
    
mediana_error = mediana['error']

#%% GRAFICO LDA PAPER VS MEJORADO VS MEDIANA
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

x = range(-100,1000,4)

y1 = (np.array(paper['media']) - 0.5)*2
y1menos = (np.array(paper['media'] - paper['error']) - 0.5)*2
y1mas = (np.array(paper['media'] + paper['error']) - 0.5)*2

plt.plot(x, y1, label ='Original LDA')
plt.fill_between(x, y1menos, y1mas, alpha=0.2)
# plt.fill_between(x, paper['media'] - paper['error'], paper['media'] + paper['error'],
#                   alpha=0.2)

y2 = (np.array(mejor['media']) - 0.5)*2
y2menos = (np.array(mejor['media'] - mejor['error']) - 0.5)*2
y2mas = (np.array(mejor['media'] + mejor['error']) - 0.5)*2

plt.plot(x, y2, label ='PO4 4-12.5 Hz LDA')
plt.fill_between(x, y2menos, y2mas, alpha=0.2)
# plt.fill_between(x, mejor['media'] - mejor['error'], mejor['media'] + mejor['error'],
#                   alpha=0.2)

y3 = (mediana - 0.5788)/0.4212
med_menos = (np.array(mediana-mediana_error) - 0.5788)/0.4212
med_mas = (np.array(mediana+mediana_error) - 0.5788)/0.4212

plt.plot(x,y3, label='Median')
plt.fill_between(x, med_menos, med_mas, alpha=0.2)

plt.xlabel('Time (ms)', fontsize=33, labelpad=30)
plt.tick_params(axis='both', labelsize=27)
plt.ylabel("Accuracy over randomness", fontsize=33, labelpad=30)
plt.legend(loc='best', fontsize=25)
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.show()
