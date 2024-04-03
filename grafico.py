import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pickle
import pandas as pd
from matplotlib import gridspec
import seaborn as sns
from PIL import Image
import math
import pingouin as pg
from scipy import stats
import scikit_posthocs as sp
from statsmodels.stats import multitest

import sys
sys.path.append('/home/ubuntuverde/Eric/rsvp/scripts/')
from corrstats import dependent_corr
# pg.compute_esci

x = range(-100,1000,4)
carpeta_figuras = '/home/ubuntuverde/Eric/rsvp/figuras/nuevas/'

def matriz(cat_con, con_cat, masgrandes):
    matrix = []
    for m in masgrandes:
        vector = np.zeros(len(masgrandes))
        for n in range(len(masgrandes)):
            if m != masgrandes[n]:
                for concepto in cat_con[m]:
                    if masgrandes[n] in con_cat[concepto]:
                        vector[n] += 1
            else:
                vector[n] = len(cat_con[m])
        # print(m, len(cat_con[m]))
        # print(vector)
        vector /= len(cat_con[m])
        matrix.append(vector*100)
    return matrix

def ic(r, n):
    stderr = 1.0 / math.sqrt(n - 3)
    delta = 1.96 * stderr
    lower = math.tanh(math.atanh(r) - delta)
    upper = math.tanh(math.atanh(r) + delta)
    return (lower, upper)

def linea(ax, arr, y, color='black'):
    '''ax:plot; arr: arreglo de significativos'''
    lista = []
    i = 0
    while i < len(arr) - 1:
        if arr[i]:
            lista.append(-100 + i*4)
            i += 1
            while i < 275 and arr[i] == True:
                lista.append(-100 + i*4)
                i += 1
            ax.plot(lista, [y]*len(lista), linewidth=4, color=color)
            lista = []
        else:
            i +=1
            
def solapamiento(lower1, upper1, lower2, upper2):
    lista = [False]*50
    for t in range(50,85):
        if lower1[t]<=lower2[t] and upper1[t]>=lower2[t]:
            lista.append(False)
        elif lower2[t]<=lower1[t] and upper2[t]>=lower1[t]:
            lista.append(False)
        else:
            lista.append(True)
    lista += [False]*190
    return lista

#%% Fig 1 datos
f_dic = '/home/ubuntuverde/Eric/rsvp/scripts/categorias/'

with open(f_dic + 'cat_con.pkl', 'rb') as f:
      cat_con = pickle.load(f)
    
with open(f_dic + 'con_cat.pkl', 'rb') as f:
      con_cat = pickle.load(f)
  
heights = pd.DataFrame([k,len(cat_con[k])/18.54] for k in cat_con.keys())
heights.columns = ['category', 'amount']
heights.sort_values(by='amount',inplace=True)
cants = list(heights['amount'])

masgrandes = list(heights['category'])[-15:]
heat = pd.DataFrame(matriz(cat_con, con_cat, masgrandes[::-1]))
heat.index = masgrandes[::-1]
heat.columns = masgrandes[::-1]

img = np.asarray(Image.open('/home/ubuntuverde/Eric/rsvp/figuras/fig1_1.png'))

#%% Fig 1 Grafico
fig = plt.figure(figsize=(30,32))
fig.suptitle('Fig. 1',x=0.1, y=0.98, size=55)

gs0 = gridspec.GridSpec(2,1, top=0.98)# right=1, bottom=-0.2) #hspace=-0.1, bottom=-0.0, top=1.1)
gs00 = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=gs0[0], wspace=0.5, width_ratios=[1, 1.3])
gs01 = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=gs0[1])

axa = fig.add_subplot(gs00[0,0])
axb = fig.add_subplot(gs00[0,1])
axc = fig.add_subplot(gs01[0,0])
ax = [axa,axb,axc]
font = 45
mostrar = 15


'''arriba izquierda'''
ax[0].set_title('a', x=-0.1, y=1, fontsize=font)
ax[0].spines['left'].set_color('none')
ax[0].spines['right'].set_color('none')
ax[0].spines['top'].set_color('none')
ax[0].spines['bottom'].set_color('none')
ax[0].tick_params(axis='both', which='both', colors='white')
ax[0].imshow(img)


'''arriba derecha'''
sns.heatmap(data=heat, ax=ax[1], cmap='YlGnBu', square=True)
ax[1].set_title('b', x=-0.25, y=1, fontsize=font)
ax[1].tick_params(axis='both', which='major', labelsize=30, width=1.5, length=10)
cbar = ax[1].collections[0].colorbar
cbar.ax.get_yaxis().labelpad = 25
cbar.ax.set_ylabel('Shared ratio (%)', fontsize=45, rotation=90, labelpad=10)
cbar.ax.tick_params(labelsize=35)


'''abajo centro'''
ax[2].set_title('c', x=-0.0, y=1.05, fontsize=font)
ax[2].spines['left'].set_position(('axes',-0.01))
ax[2].spines['left'].set_color('black')
ax[2].spines['right'].set_color('none')
ax[2].yaxis.tick_left()
ax[2].spines['bottom'].set_position(('axes',0))
ax[2].spines['bottom'].set_color('black')
ax[2].spines['top'].set_color('none')
ax[2].xaxis.tick_bottom()
ax[2].margins(x=0)
ax[2].bar(x=range(len(cat_con.keys())),height=cants[::-1])
ax[2].plot(range(len(cat_con.keys())),[10/18.54]*len(cat_con.keys()), 
           color='red', linestyle='dashed', linewidth=8)
ax[2].set_ylabel('Concepts ratio (%)', fontsize=50, labelpad=30)
ax[2].set_xlabel('Category #',  fontsize=50, labelpad=30)
# ax[2].yaxis.set_label_position("right")
# ax[2].yaxis.tick_right()
ax[2].tick_params(axis='both', labelsize=30)

axi = ax[2].inset_axes([0.3, 0.2, 0.65, 0.8])
axi.margins(y=0.01)
axi.barh(range(mostrar),cants[-mostrar:])
axi.set_yticks(range(mostrar), labels=list(heights['category'])[-mostrar:], size=35)
axi.set_xlabel('Concepts ratio (%)',  fontsize=45)
# axi.set_ylabel('Category #')
# axi.invert_yaxis()  # labels read top-to-bottom
axi.plot([10/18.54]*mostrar, range(mostrar), color='red', linestyle='dashed', linewidth=9)
# axi.tick_params(labelleft=True, labelbottom=True)
axi.tick_params(axis='both', which='major', labelsize=35)

plt.savefig(carpeta_figuras + 'fig1.png', edgecolor='black', dpi=600, facecolor='white')

#%% Fig 2.5 datos
#archivo: mejor_canal_filtro_causal
#MUESTRA DE FILTROS, SUJETO 38
path = '/home/ubuntuverde/Eric/rsvp/scripts/mejor_canal/'
with open(path + 'filtros.pkl', 'rb') as f:    #prueba grande
    filtros = pickle.load(f)  
fases = ['zero', 'minimum']   
    
path2 = '/home/ubuntuverde/Eric/rsvp/scripts/mediana/'
with open(path2 + 'dic_lda.pkl', 'rb') as f:    #prueba grande
    lda = pickle.load(f)  
    
#%% Fig 2.5 grafico
fig, axs = plt.subplots(3, 1)
fig.set_size_inches(10, 6)

axs[0].plot(x, lda[38], c="red", label='LDA')
axs[1].plot(x, filtros['zero'], c='blue', label='zero')
axs[2].plot(x, filtros['minimum'], c="green", label='minimum')
for k in range(3):
    leg = axs[k].legend(loc='best', fontsize=25, frameon=False, ncol=2, handlelength=0)
    
axs[0].axvline(x[np.argmax(filtros['zero'])],ymin=-1.2,ymax=1,c="blue",linewidth=2,zorder=0, clip_on=False, linestyle='--')
axs[0].axvline(x[54],ymin=0,ymax=1,c="red",linewidth=2, zorder=0,clip_on=False, linestyle='--')
axs[0].axvline(x[np.argmax(filtros['minimum'])],ymin=-1.2,ymax=1,c="green",linewidth=2,zorder=0, clip_on=False, linestyle='--')

axs[1].axvline(x[np.argmax(filtros['zero'])],ymin=-1.2,ymax=1,c="blue",linewidth=2,zorder=0, clip_on=False, linestyle='--')
axs[1].axvline(x[np.argmax(filtros[f])],ymin=-1.2,ymax=1,c="green",linewidth=2,zorder=0, clip_on=False, linestyle='--')
axs[1].axvline(x[54],ymin=0,ymax=1.2,c="red",linewidth=2, zorder=0,clip_on=False, linestyle='--')

axs[2].axvline(x[np.argmax(filtros['zero'])],ymin=0,ymax=1,c="blue",linewidth=2,zorder=0, clip_on=False, linestyle='--')
axs[2].axvline(x[np.argmax(filtros['minimum'])],ymin=0,ymax=1,c="green",linewidth=2,zorder=0, clip_on=False, linestyle='--')
axs[2].axvline(x[54],ymin=0,ymax=1.2,c="red",linewidth=2,zorder=0, clip_on=False, linestyle='--')

for k in range(3):
    axs[k].spines['right'].set_color('none')
    axs[k].yaxis.tick_left()
    
    # set the y-spine
    axs[k].spines['bottom'].set_position(('axes',0))
    axs[k].spines['bottom'].set_color('none')
    
    # turn off the top spine/ticks
    axs[k].spines['top'].set_color('none')

axs[2].spines['bottom'].set_position(('axes',0))
axs[2].spines['bottom'].set_color('black')

axs[0].tick_params(axis='x', bottom=False, labelbottom=False)
axs[1].tick_params(axis='x', bottom=False, labelbottom=False)
axs[2].tick_params(axis='x', labelsize=15)

axs[2].set_xlabel('Time (ms)', fontsize=23, labelpad=20)
# axs[0].set_ylabel("Cohen's d ", fontsize=35, labelpad=30)
# axs[1].set_ylabel("Accuracy over chance",  fontsize=35, labelpad=30)
plt.show()

#%% Fig 3 datos
#archivo: mejor_canal
    
path = '/home/ubuntuverde/Eric/rsvp/scripts/mejor_canal/'
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

with open(path + 'dic_cohen_canales.pkl', 'rb') as f:
    dic_cohen_canales= pickle.load(f)  
    
with open(path + 'lda_mejorado.pkl', 'rb') as f:
    lda_mejor = pickle.load(f) 

with open(path + 'lda_paper.pkl', 'rb') as f:
    paper = pickle.load(f)  
    
 
path_mediana = '/home/ubuntuverde/Eric/rsvp/scripts/mediana/dic_mediana_chica.pkl'
with open(path_mediana, 'rb') as f:
    dic_mediana= pickle.load(f)  
    
buenos = ['PO4', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'POz', 'PO8', 'P2']

path2 = '/home/ubuntuverde/Eric/rsvp/scripts/mediana/'
sujetos = set(range(1,51)) - {6}

with open(path2 + 'dic_lda.pkl', 'rb') as f:    #prueba grande
    lda = pickle.load(f)  

with open(path2 + 'dic_mediana.pkl', 'rb') as f:    #prueba grande
    med = pickle.load(f)  
    
#solo considero válidos de la muestras 50 a 85
sig = [False]*50
sig += [stats.ranksums([(lda[i][t]-0.5)*2 for i in sujetos], 
                 [(med[i][t-25]- 0.5788)/0.4212 for i in sujetos])[1] < 0.01 for t in range(50,85)]
sig += [False]*190
# truesig = [(s*4)-25 for s in range(275) if sig[s]] 
  
#el pico está en la muestra 56

#hago kruskal para la figura C
# sujetos_chico = set(range(2,51)) - {6}
# for t in range(50,85):
#     res = stats.kruskal([paper[s][t] for s in sujetos_chico],[lda_mejor[s][t] for s in sujetos_chico],
#                         [dic_mediana[s][t] for s in sujetos_chico])[1] < 0.05
#     print(res)  #dan todos significativos
    
# for t in range(50,85):
#     res = sp.posthoc_dunn([[paper[s][t] for s in sujetos_chico],[lda_mejor[s][t] for s in sujetos_chico],
#                         [dic_mediana[s][t] for s in sujetos_chico]])[1][1:] < 0.05
    # print(t, res, sep='\n')  #dan todos significativos    
#de 51 a 85 da significativo todos vs todos

#%% Fig 3 gráfico
fig, axs = plt.subplots(2, 2)
fig.set_size_inches(27, 17)

# set the x-spine (see below for more info on `set_position`)
for i in range(2):
    for j in range(2):
        axs[i,j].spines['left'].set_position(('axes',0))
        axs[i,j].spines['left'].set_color('black')
        
        # turn off the right spine/ticks
        axs[i,j].spines['right'].set_color('none')
        axs[i,j].yaxis.tick_left()
        
        # set the y-spine
        axs[i,j].spines['bottom'].set_position(('axes',0))
        axs[i,j].spines['bottom'].set_color('black')
        
        # turn off the top spine/ticks
        axs[i,j].spines['top'].set_color('none')
        axs[i,j].xaxis.tick_bottom()
        axs[i,j].margins(x=0)
        axs[i,j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
# fig.suptitle('Fig. 3', x=0.06, y=1.02, size=55)
# fig.tight_layout(pad=4.0)
                        
'''PANEL ARRIBA IZQUIERDA'''
media = dic_cohen_canales['media']
error = dic_cohen_canales['error']

for b in buenos:
    axs[0,0].plot(x, media[b], label=b, linewidth=3)
    axs[0,0].fill_between(x, media[b] - error[b], media[b] + error[b],
                       alpha=0.15)
leg = axs[0,0].legend(loc='best', fontsize=30, frameon=False, ncol=2, handlelength=1)
axs[0,0].text(-10,0.42,'PO4', fontsize=30)
axs[0,0].set_title('a', x=-0.1, y=1.1, fontsize=45)

# set the linewidth of each legend object
for legobj in leg.legend_handles:
    legobj.set_linewidth(4.0)

'''PANEL ARRIBA DERECHA'''
media = dic_cohen_po4['media']
error = dic_cohen_po4['error']
axs[0, 1].plot(x, media, label='PO4 broadband', color='olive')
axs[0, 1].fill_between(x, media - error, media + error,
                  color='olive', alpha=0.2)

media = dic_cohen_mejor['media']
error = dic_cohen_mejor['error']
axs[0, 1].plot(x, media, label='PO4 4-12.5 Hz', color='purple')
axs[0, 1].fill_between(x, media - error, media + error,
                  color='purple', alpha=0.2)

media = dic_po4_4_8['medias']
error = dic_po4_4_8['error']
axs[0, 1].plot(x, media, label='PO4 4-8 Hz', color='cyan')
axs[0, 1].fill_between(x, media - error, media + error,
                  color='cyan', alpha=0.2)

media = dic_po4_4_30['medias']
error = dic_po4_4_30['error']
axs[0, 1].plot(x, media, label='PO4 4-30 Hz', color='orange')
axs[0, 1].fill_between(x, media - error, media + error,
                  color='orange', alpha=0.2)

#todos dan significativo vs broadband a los 124 ms (ANOVA, Tukey)
sig = [False]*50 + [True]*35 + [False]*190
linea(axs[0,1],sig,0.83)
axs[0,1].text(x=150, y=0.85, s='*', fontsize=40)

leg = axs[0, 1].legend(loc='best', fontsize=30, frameon=False, handlelength=1)
axs[0,1].set_title('b', x=-0.1, y=1.1, fontsize=45)

for legobj in leg.legend_handles:
    legobj.set_linewidth(4.0)

'''PANEL ABAJO IZQUIERDA'''
y1 = (np.array(paper['media']) - 0.5)*2
y1menos = (np.array(paper['media'] - paper['error']) - 0.5)*2
y1mas = (np.array(paper['media'] + paper['error']) - 0.5)*2
axs[1,0].plot(x, y1, label ='Original LDA', color='red')
axs[1,0].fill_between(x, y1menos, y1mas, alpha=0.2, color='red')

y2 = (np.array(lda_mejor['media']) - 0.5)*2
y2menos = (np.array(lda_mejor['media'] - lda_mejor['error']) - 0.5)*2
y2mas = (np.array(lda_mejor['media'] + lda_mejor['error']) - 0.5)*2
axs[1,0].plot(x, y2, label ='PO4 4-12.5 Hz LDA', color='orange')
axs[1,0].fill_between(x, y2menos, y2mas, alpha=0.2, color='orange')

y3 = (dic_mediana['media'] - 0.5788)/0.4212
med_menos = (np.array(dic_mediana['media']-dic_mediana['error']) - 0.5788)/0.4212
med_mas = (np.array(dic_mediana['media']+dic_mediana['error']) - 0.5788)/0.4212
axs[1,0].plot(x,y3, label='PO4 4-12.5 Hz Median', color='blue')
axs[1,0].fill_between(x, med_menos, med_mas, alpha=0.2, color='blue')
axs[1,0].plot(x,np.zeros(275),color='grey')

leg = axs[1,0].legend(loc='best', fontsize=30, frameon=False, handlelength=1)
axs[1,0].set_title('c', x=-0.1, fontsize=45)
for legobj in leg.legend_handles:
    legobj.set_linewidth(4.0)

sig = [False]*51 + [True]*34 + [False]*190
linea(axs[1,0],sig,0.23)
axs[1,0].text(x=130, y=0.24, s='**', fontsize=40)

'''PANEL ABAJO DERECHA'''
media = lda['media']
y1 = (np.array(media) - 0.5)*2
y1menos = ((np.array(media) - lda['error']) - 0.5)*2
y1mas = ((np.array(media) + lda['error']) - 0.5)*2

axs[1,1].plot(x, y1, label ='Original LDA', color='red')
axs[1,1].fill_between(x, y1menos, y1mas, alpha=0.2, color='red')

y2 = (np.array(med['media']) - 0.5788)/0.4212
y2menos = (np.array(med['media'] - med['error'])  - 0.5788)/0.4212
y2mas = (np.array(med['media'] + med['error'])  - 0.5788)/0.4212

axs[1,1].plot(x,y2, label='PO4 4-12.5 Hz Median', color='blue')
axs[1,1].fill_between(x, y2menos, y2mas, alpha=0.2, color='blue')
leg = axs[1,1].legend(loc='best', fontsize=30, frameon=False, handlelength=1)
axs[1,1].plot(x, [0]*275, color='grey')
axs[1,1].set_title('d', x=-0.1, fontsize=45)
linea(axs[1,1], sig, 0.09)
axs[1,1].text(x=130, y=0.095, s='**', fontsize=40)
for legobj in leg.legend_handles:
    legobj.set_linewidth(4.0)

axs[0,0].tick_params(axis='x', which='both', labelbottom=False)
axs[0,0].tick_params(axis='y', which='both', labelsize=25)
axs[0,1].tick_params(axis='x', which='both', labelbottom=False)
axs[0,1].tick_params(axis='y', which='both', labelsize=25)
axs[1,0].tick_params(axis='both', labelsize=25)
axs[1,1].tick_params(axis='both', labelsize=25)
axs[1,0].set_xlabel('Time (ms)', fontsize=35, labelpad=30)
axs[1,1].set_xlabel('Time (ms)', fontsize=35, labelpad=30)
axs[0,0].set_ylabel("Cohen's d ", fontsize=35, labelpad=30)
axs[1,0].set_ylabel("Accuracy over chance",  fontsize=35, labelpad=30)

# for ax in axs.flat:
#     ax.set(xlabel='Time (ms)', ylabel="Cohen's d")

# Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()
    
#     ax.set()

#%% Fig 4 datos
#archivo: voltajes_crudos_luminancia, spearman pca 1_cp
path4 = '/home/ubuntuverde/Eric/rsvp/scripts/pca/'
with open(path4 + 'spearman_sujeto.pkl', 'rb') as f:    #22248 valores
    muestras_sujeto = pickle.load(f)

#cargar sujetos, lda y mediana de la figura 3
muestras_sujeto_1 = muestras_sujeto['cp1']
muestras_sujeto_2 = muestras_sujeto['cp2']

with open(path4 + 'dic_rho_p.pkl', 'rb') as f:
    rho = pickle.load(f) 
       
df = pd.read_csv(path4 + 'df_radar.csv') 
df.columns = ['OE 6', 'OE 14', 'SFE 5', 'SFE 6', 'SFE 7', 'Luminance', 'Green', 'Blue']

#primero tengo que hacer una transformación de Fisher
#b y c
sig,sig2 = [], []
for t in range(50,85):
    arc = np.arctanh([muestras_sujeto['cp1'][s][t] for s in sujetos])
    sig.append(stats.ttest_1samp(arc,0)[1] < 0.05)
    arc2 = np.arctanh([muestras_sujeto['cp2'][s][t] for s in sujetos])
    sig2.append(stats.ttest_1samp(arc2,0)[1] < 0.05)
sig = [False]*50 + sig + [False]*190
sig2 = [False]*50 + sig2 + [False]*190

#correlacion entre LDA y mediana
corr_lda_med = [stats.spearmanr([lda[s][t] for s in sujetos], [med[s][t] for s in sujetos])[0] for t in range(275)]

img = np.asarray(Image.open('/home/ubuntuverde/Eric/rsvp/figuras/representativas.png'))

#%% Fig 4 grafico
fig = plt.figure(figsize=(40, 18))
# fig.suptitle('Fig. 4', x=0.04, y=1.02, size=55)

gs0 = gridspec.GridSpec(1, 2, wspace=0.1, left=0, right=0.9)
gs00 = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=gs0[0])
gs01 = gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec=gs0[1], wspace=0.3)

# ax01 = fig.add_subplot(gs00[0,0])
categories = df.columns
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
ax = fig.add_subplot(gs00[0,0], polar=True)

ax.set_title('a', x=0, fontsize=45)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize = 35)
 # 'horizontalalignment': loc})
 
ax.set_rlabel_position(90)
ax.set_yticklabels(labels=[-0.1,0,0.1,0.2,0.3,0.4,0.5], size=35)
nombres = ['PCA 1', 'PCA 2']
for n,row in enumerate(df.iloc):
    values = row.values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=3, linestyle = 'solid', 
            label=nombres[n]) # Ploteamos la línea
    leg = ax.legend(loc=[0.7,1],fontsize=45, frameon=False, handlelength=1)
for legobj in leg.legend_handles:
        legobj.set_linewidth(5.0)
    
axsb = fig.add_subplot(gs01[0,0])
axsb.set_title('b', x=-0.1, y=1.1, fontsize=45)
axsc = fig.add_subplot(gs01[0,1])
axsc.set_title('d', x=-0.1, y=1.1, fontsize=45)
axsd = fig.add_subplot(gs01[1,0])
axsd.set_title('c', x=-0.1, fontsize=45)
axse = fig.add_subplot(gs01[1,1])
axse.set_title('e', x=-0.1, fontsize=45)

axs = [axsb,axsc,axsd,axse]
for i in range(4):
    axs[i].spines['left'].set_position(('axes',0))
    axs[i].spines['left'].set_color('black')
    
    # turn off the right spine/ticks
    axs[i].spines['right'].set_color('none')
    axs[i].yaxis.tick_left()
    
    # set the y-spine
    axs[i].spines['bottom'].set_position(('axes',0))
    axs[i].spines['bottom'].set_color('black')
    
    # turn off the top spine/ticks
    axs[i].spines['top'].set_color('none')
    axs[i].xaxis.tick_bottom()
    axs[i].margins(x=0)
    axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[i].set_xticks([0, 200,400,600,800])
    
#agrego los intervalos de confianza de las spearman
'''ARRIBA IZQUIERDA PRIMERA COMPONENTE'''  
axs[0].plot(x, muestras_sujeto_1['media'])
lowers0, uppers0 = pg.compute_esci(stat=muestras_sujeto_1['media'], nx=22248, ny=22248, eftype='r', decimals=3)

axs[0].fill_between(x, lowers0, uppers0, color='black', alpha=0.2)

axs[0].tick_params(axis='both', which='major', labelsize=30)
# axs[0].fill_between(x, muestras_sujeto_1['media'] - muestras_sujeto_1['error'], 
#               muestras_sujeto_1['media'] + muestras_sujeto_1['error'], color='black', alpha=0.2)

axs[0].plot(x,np.zeros(275),color='grey')
axs[0].locator_params(axis='y', nbins=5)
linea(axs[0],sig, 0.27)

'''ABAJO IZQUIERDA: SEGUNDA COMPONENTE'''
axs[2].plot(x, muestras_sujeto_2['media'])
lowers2, uppers2 = pg.compute_esci(stat=muestras_sujeto_2['media'], nx=22248, ny=22248, eftype='r', decimals=3)

axs[2].fill_between(x, lowers2, uppers2, color='black', alpha=0.2)

axs[2].tick_params(axis='both', which='major', labelsize=30)
# axs[2].fill_between(x, muestras_sujeto_2['media'] - muestras_sujeto_2['error'], 
#               muestras_sujeto_2['media'] + muestras_sujeto_2['error'], color='black', alpha=0.2)
axs[2].plot(x,np.zeros(275),color='grey')
axs[2].locator_params(axis='y', nbins=5)
linea(axs[2],sig2, 0.1)


'''ARRIBA DERECHA: PRIMERA COMPONENTE LDA Y MEDIANA'''

y1 = np.array(rho['LDA 1CP']['rho'])
axs[1].plot(x, y1, color="red", label='LDA')
lowers1lda,uppers1lda = pg.compute_esci(stat=y1, nx=1717731, ny=1717731, eftype='r', decimals=3)
axs[1].fill_between(x, lowers1lda, uppers1lda, color='red', alpha=0.2)

y2 = np.array(rho['Mediana 1CP']['rho'])
axs[1].plot(x, y2, color="blue", label='Median')
lowers1med,uppers1med = pg.compute_esci(stat=y2, nx=1717731, ny=1717731, eftype='r', decimals=3)
axs[1].fill_between(x, lowers1med, uppers1med, color='blue', alpha=0.2)

leg = axs[1].legend(loc='best', fontsize=35, frameon=False, handlelength=1)
axs[1].plot(x,np.zeros(275),color='grey')
axs[1].locator_params(axis='y', nbins=5)
for legobj in leg.legend_handles:
        legobj.set_linewidth(5.0)

p_lda,p_med = [], []
for t in range(50,85):
    p_lda.append(rho['LDA 1CP']['p'][t] < 0.05)
    p_med.append(rho['Mediana 1CP']['p'][t] < 0.05)
p_lda = [False]*50 + p_lda + [False]*190
p_med = [False]*50 + p_med + [False]*190
linea(axs[1], p_lda, 0.59, color='red')
linea(axs[1], p_med, 0.61, color='blue')
linea(axs[1], p_med, 0.63, color='black')
# for t in range(50,85):  #todos significativos
#     print(dependent_corr(rho['LDA 1CP']['rho'][t],rho['Mediana 1CP']['rho'][t], 
#                          corr_lda_med[t], n=1717731))

'''ABAJO DERECHA: SEGUNDA COMPONENTE LDA Y MEDIANA'''
y1 = np.array(rho['LDA 2CP']['rho'])
axs[3].plot(x, y1, color="red", label='LDA')
lowers3lda,uppers3lda = pg.compute_esci(stat=y1, nx=1717731, ny=1717731, eftype='r', decimals=3)
axs[3].fill_between(x, lowers3lda, uppers3lda, color='red', alpha=0.2)

y2 = np.array(rho['Mediana 2CP']['rho'])
axs[3].plot(x, y2, color="blue", label='Median')
lowers3med,uppers3med = pg.compute_esci(stat=y2, nx=1717731, ny=1717731, eftype='r', decimals=3)
axs[3].fill_between(x, lowers3med, uppers3med, color='blue', alpha=0.2)

leg = axs[3].legend(loc='best', fontsize=35, frameon=False, handlelength=1)
axs[3].plot(x,np.zeros(275),color='grey')
axs[3].locator_params(axis='y', nbins=5)
for legobj in leg.legend_handles:
        legobj.set_linewidth(5.0)

p_lda,p_med,p_ldamed = [], [], []
for t in range(50,85):
    p_lda.append(rho['LDA 2CP']['p'][t] < 0.05)
    p_med.append(rho['Mediana 2CP']['p'][t] < 0.05)
    p_ldamed.append(dependent_corr(rho['LDA 2CP']['rho'][t],rho['Mediana 2CP']['rho'][t], 
                         corr_lda_med[t], n=1717731)[1] <0.05)
p_lda = [False]*50 + p_lda + [False]*190
p_med = [False]*50 + p_med + [False]*190
p_ldamed = [False]*50 + p_ldamed + [False]*190

linea(axs[3], p_lda, 0.12, color='red')
linea(axs[3], p_med, 0.125, color='blue')
linea(axs[3], p_ldamed, 0.13, color='black')
    
axs[0].tick_params(axis='x', which='both', labelbottom=False)
axs[0].tick_params(axis='y', which='both', labelsize=30)
axs[0].set_ylabel("Spearman's' correlation", fontsize=38, labelpad=30)
axs[2].set_xlabel('Time (ms)', fontsize=38, labelpad=30)
axs[2].set_ylabel("Spearman's' correlation", fontsize=38, labelpad=29)
axs[2].tick_params(axis='both', labelsize=30)
axs[1].tick_params(axis='x', which='both', labelbottom=False)
axs[1].tick_params(axis='y', which='both', labelsize=30)
axs[3].tick_params(axis='y', which='both', labelsize=30)
axs[3].tick_params(axis='x', which='both', labelsize=30)
axs[3].set_xlabel('Time (ms)', fontsize=38, labelpad=30)

# ax4 = fig.add_subplot(2,3,1)
# ax4.imshow(img)

plt.show()
# plt.savefig(carpeta_figuras + 'fig4.png', dpi=800, facecolor='white')

#%% Fig 5 datos
#archivo: primer_cuartil_pca
folder = '/home/ubuntuverde/Eric/rsvp/prueba_grande/cuartil_pca/'

with open(folder + 'lum_lda_cohen_cuartil.pkl', 'rb') as f:
    lda_cohen = pickle.load(f)

with open(folder + 'lum_med_cohen_cuartil.pkl', 'rb') as f:
    med_cohen = pickle.load(f)
    
with open(folder + 'lum_lda_spearman_cuartil.pkl', 'rb') as f:
    lda_spearman = pickle.load(f)
        
with open(folder + 'lum_med_spearman_cuartil.pkl', 'rb') as f:
    med_spearman = pickle.load(f)

with open(folder + 'lda_med_spearman.pkl', 'rb') as f:
    lda_med_spearman = pickle.load(f)
    
#%% Fig 5 Grafico
fig, axs = plt.subplots(2, 2)
fig.set_size_inches(25, 17)

axs[0,0].set_title('a', x=-0.1, y=1.1, fontsize=45)
axs[0,1].set_title('b', x=-0.1, y=1.1, fontsize=45)
axs[1,0].set_title('c', x=-0.1, fontsize=45)
axs[1,1].set_title('d', x=-0.1, fontsize=45)


for i in range(2):
    for j in range(2):
        axs[i,j].spines['left'].set_position(('axes',0))
        axs[i,j].spines['left'].set_color('black')
        
        # turn off the right spine/ticks
        axs[i,j].spines['right'].set_color('none')
        axs[i,j].yaxis.tick_left()
        
        # set the y-spine
        axs[i,j].spines['bottom'].set_position(('axes',0))
        axs[i,j].spines['bottom'].set_color('black')
        
        # turn off the top spine/ticks
        axs[i,j].spines['top'].set_color('none')
        axs[i,j].xaxis.tick_bottom()
        axs[i,j].margins(x=0)
        axs[i,j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# fig.suptitle('Fig. 5', x=0.055, y=1.02, size=55)



'''ARRIBA IZQUIERDA: PRIMERA COMPONENTE LDA Y MEDIANA'''  
axs[0,0].plot(x, lda_cohen[0], color='red', label='LDA')
axs[0,0].plot(x, med_cohen[0], color='blue', label='Median')
axs[0,0].tick_params(axis='both', which='major', labelsize=27)
axs[0,0].plot(x,np.zeros(275),color='grey')
lower0lda,upper0lda = pg.compute_esci(stat=lda_cohen[0], nx=107416, ny=1610315, eftype='cohen', decimals=3)
axs[0,0].fill_between(x, lower0lda, upper0lda, color='red', alpha=0.2)
lower0med,upper0med = pg.compute_esci(stat=med_cohen[0], nx=107416, ny=1610315, eftype='cohen', decimals=3)
axs[0,0].fill_between(x, lower0med, upper0med, color='blue', alpha=0.2)

sig_lda0 = [False]*50 + lda_cohen['sig0'][50:85] + [False]*190
linea(axs[0,0],sig_lda0, y=0.85, color='red')
sig_med0 = [False]*50 + med_cohen['sig0'][50:85] + [False]*190
linea(axs[0,0],sig_med0, y=0.875, color='blue')
linea(axs[0,0],solapamiento(lower0lda, upper0lda, lower0med,upper0med), y=0.9, color='black') #no se solapan los IC

leg = axs[0,0].legend(loc='best', fontsize=50, frameon=False, handlelength=0.5)
for legobj in leg.legend_handles:
    legobj.set_linewidth(8.0)

'''ARRIBA DERECHA: SEGUNDA COMPONENTE LDA Y MEDIANA'''
axs[0,1].plot(x, lda_cohen[1], color='red', label='LDA')
axs[0,1].plot(x, med_cohen[1], color='blue', label='Median')
axs[0,1].tick_params(axis='both', which='major', labelsize=27)
axs[0,1].plot(x,np.zeros(275),color='grey')
lower1lda,upper1lda = pg.compute_esci(stat=lda_cohen[1], nx=107416, ny=1610315, eftype='cohen', decimals=3)
axs[0,1].fill_between(x, lower1lda, upper1lda, color='red', alpha=0.2)
lower1med,upper1med = pg.compute_esci(stat=med_cohen[1], nx=107416, ny=1610315, eftype='cohen', decimals=3)
axs[0,1].fill_between(x, lower1med, upper1med, color='blue', alpha=0.2)

sig_lda1 = [False]*50 + lda_cohen['sig1'][50:85] + [False]*190
linea(axs[0,1],sig_lda1, y=0.211, color='red')
sig_med1 = [False]*50 + med_cohen['sig1'][50:85] + [False]*190
linea(axs[0,1],sig_med1, y=0.22, color='blue')
sig_ldamed = [False]*50 + [True]*35 + [False]*190    #no se solapan los IC
linea(axs[0,1],solapamiento(lower1lda, upper1lda, lower1med,upper1med), y=0.23, color='black') #no se solapan los IC


'''ABAJO IZQUIERDA: PRIMERA COMPONENTE SPEARMAN'''
axs[1,0].plot(x, [lda_spearman[0][t][0] for t in range(275)], color='red', label='LDA')
lowerlda0,upperlda0 = pg.compute_esci(stat=[lda_spearman[0][t][0] for t in range(275)], nx=1717731, ny=1717731, eftype='r', decimals=3)
axs[1,0].fill_between(x, lowerlda0, upperlda0, color='red', alpha=0.2)

axs[1,0].plot(x, [med_spearman[0][t][0] for t in range(275)], color='blue', label='Median')
lowermed0,uppermed0 = pg.compute_esci(stat=[med_spearman[0][t][0] for t in range(275)], nx=1717731, ny=1717731, eftype='r', decimals=3)
axs[1,0].fill_between(x, lowermed0, uppermed0, color='blue', alpha=0.2)

axs[1,0].tick_params(axis='both', which='major', labelsize=27)
axs[1,0].plot(x,np.zeros(275),color='grey')

sig_lda0 = [False]*50 + [lda_spearman[0][t][1]<0.05 for t in range(50,85)] + [False]*190
linea(axs[1,0],sig_lda0, y=0.055, color='red')
sig_med0 = [False]*50 + [med_spearman[0][t][1]<0.05 for t in range(50,85)] + [False]*190
linea(axs[1,0],sig_med0, y=0.06, color='blue')
linea(axs[1,0],sig_med0, y=0.065, color='black')
# for t in range(50,85): #son todos significativos
#     clm = dependent_corr(lda_spearman[0][t][0], med_spearman[0][t][0], lda_med_spearman[t], 1717731)
#     print(clm[1])

'''ABAJO DERECHA: SEGUNDA COMPONENTE SPEARMAN'''
axs[1,1].plot(x, [lda_spearman[1][t][0] for t in range(275)], color='red', label='LDA')
lowerlda1,upperlda1 = pg.compute_esci(stat=[lda_spearman[1][t][0] for t in range(275)], nx=1717731, ny=1717731, eftype='r', decimals=3)
axs[1,1].fill_between(x, lowerlda1, upperlda1, color='red', alpha=0.2)

axs[1,1].plot(x, [med_spearman[1][t][0] for t in range(275)], color='blue', label='Median')
lowermed1,uppermed1 = pg.compute_esci(stat=[med_spearman[1][t][0] for t in range(275)], nx=1717731, ny=1717731, eftype='r', decimals=3)
axs[1,1].fill_between(x, lowermed1, uppermed1, color='blue', alpha=0.2)

axs[1,1].tick_params(axis='both', which='major', labelsize=27)
axs[1,1].plot(x,np.zeros(275),color='grey')

sig_lda1 = [False]*50 + [lda_spearman[1][t][1]<0.05 for t in range(50,85)] + [False]*190
linea(axs[1,1],sig_lda1, y=0.038, color='red')
sig_med1 = [False]*50 + [med_spearman[1][t][1]<0.05 for t in range(50,85)] + [False]*190
linea(axs[1,1],sig_med1, y=0.04, color='blue')
clm = [dependent_corr(lda_spearman[1][t][0], med_spearman[1][t][0], lda_med_spearman[t], 1717731)[1]<0.05 for t in range(50,85)]
sig = [False]*50 + clm + [False]*190
linea(axs[1,1],sig, y=0.042, color='black')

axs[0,0].tick_params(axis='x', which='both', labelbottom=False)
axs[0,0].tick_params(axis='y', which='both', labelsize=25)
axs[0,0].set_ylabel("Cohen's d", fontsize=35, labelpad=30)
axs[0,0].locator_params(axis='y', nbins=6)
axs[1,0].set_xlabel('Time (ms)', fontsize=35, labelpad=30)
axs[1,0].set_ylabel("Spearman's' correlation", fontsize=35, labelpad=25)
axs[1,0].tick_params(axis='both', labelsize=25)
axs[1,0].locator_params(axis='y', nbins=6)
axs[0,1].tick_params(axis='x', which='both', labelbottom=False)
axs[0,1].tick_params(axis='y', which='both', labelsize=25)
axs[0,1].locator_params(axis='y', nbins=5)
axs[1,1].tick_params(axis='y', which='both', labelsize=25)
axs[1,1].tick_params(axis='x', which='both', labelsize=25)
axs[1,1].set_xlabel('Time (ms)', fontsize=35, labelpad=30)
axs[1,1].locator_params(axis='y', nbins=6)

# plt.savefig(carpeta_figuras + 'fig5.png', edgecolor='black', dpi=800, facecolor='white', transparent=True)

#%% Fig 6 Datos
#archivo: clustering_jerarquico_4cl_acotado, clustering_4cl_azar
df_clusters = pd.read_csv('/home/ubuntuverde/Eric/rsvp/scripts/pca/df_4clusters.csv')
f_dic = '/home/ubuntuverde/Eric/rsvp/scripts/clustering/'
       
with open(f_dic + 'lum_lda_cohen.pkl', 'rb') as f:
    lum_lda_cohen = pickle.load(f)
with open(f_dic + 'lum_med_cohen.pkl', 'rb') as f:
    lum_med_cohen = pickle.load(f)
with open(f_dic + 'lum_lda_spearman.pkl', 'rb') as f:
    lum_lda_spearman = pickle.load(f)
with open(f_dic + 'lum_med_spearman.pkl', 'rb') as f:
    lum_med_spearman = pickle.load(f)
with open(f_dic + 'lda_med_spearman.pkl', 'rb') as f:
    lda_med_spearman = pickle.load(f)
with open(f_dic + 'clusters_cohen_lda_med_azar.pkl', 'rb') as f:
    clusters_cohen_lda_med = pickle.load(f)
    
#clusters 0:575; 1:762; 2:287; 3:230
    
#%% Fig 6 Grafico
fig = plt.figure(figsize=(45, 25))
# fig.suptitle('Fig. 6', x=0.03, y=1.01, size=65)
# 
gs0 = gridspec.GridSpec(1,2, left=0, wspace=0.2, right=1) #), hspace=-0.1, left=-0.1, bottom=-0.0, top=1.1)
gs00 = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=gs0[0])
gs01 = gridspec.GridSpecFromSubplotSpec(4,2, subplot_spec=gs0[1], hspace=0.4, wspace=0.4)

# ax01 = fig.add_subplot(gs00[0,0])
categories = df_clusters.columns
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
ax = fig.add_subplot(gs00[0,0], polar=True)

ax.set_title('a', x=0.1, y=1.05, fontsize=55)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize = 35)
 # 'horizontalalignment': loc})
 
ax.set_rlabel_position(90)
ax.set_yticklabels(labels=[-0.5,0,0.5,1,1.5], size=35)
nombres = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
colors = ['darkgreen', 'brown', 'hotpink', 'navy']
for n,row in enumerate(df_clusters.iloc):
    values = row.values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=3, linestyle = 'solid', 
            label=nombres[n], color=colors[n]) # Ploteamos la línea
    leg = ax.legend(fontsize=45, frameon=False, loc=[0.12,-0.15], ncol=2, handlelength=1)
    for legobj in leg.legend_handles:
        legobj.set_linewidth(6.0)
    
font = 55
axb = fig.add_subplot(gs01[0,0])
axb.set_title('b', x=-0.1, y=1.2, fontsize=font)
axc = fig.add_subplot(gs01[1,0])
axc.set_title('c', x=-0.1, y=1.2, fontsize=font)
axd = fig.add_subplot(gs01[2,0])
axd.set_title('d', x=-0.1, y=1.1, fontsize=font)
axe = fig.add_subplot(gs01[3,0])
axe.set_title('e', x=-0.1, y=1.1, fontsize=font)
axf = fig.add_subplot(gs01[0,1])
axf.set_title('f', x=-0.1, y=1.2, fontsize=font)
axg = fig.add_subplot(gs01[1,1])
axg.set_title('g', x=-0.1, y=1.2, fontsize=font)
axh = fig.add_subplot(gs01[2,1])
axh.set_title('h', x=-0.1, y=1.1, fontsize=font)
axi = fig.add_subplot(gs01[3,1])
axi.set_title('i', x=-0.1, fontsize=font)

axs = [axb, axc, axd, axe, axf, axg, axh, axi]

for i in range(8):
    axs[i].spines['left'].set_position(('axes',0))
    axs[i].spines['left'].set_color('black')
    
    # turn off the right spine/ticks
    axs[i].spines['right'].set_color('none')
    axs[i].yaxis.tick_left()
    
    # set the y-spine
    # axs[i].spines['bottom'].set_position(('axes',0))
    # axs[i].spines['bottom'].set_color('black')
    
    # turn off the top spine/ticks
    axs[i].spines['top'].set_color('none')
    # axs[i].xaxis.tick_bottom()
    axs[i].margins(x=0)
    axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    

alfa = 0.1
ns = [230, 575, 762, 287]

for cl in [0,1,2,3]:
    '''COHEN/LDA'''
    
    # x1 = clusters_cohen_lda_med['lda'][cl]    
    # x2 = clusters_cohen_lda_med['med'][cl]

    axs[cl].plot(x, lum_lda_cohen[cl], color='red', label='LDA')
    low1,up1 = pg.compute_esci(stat=lum_lda_cohen[cl], 
                               nx=(ns[cl]*ns[cl]-1)/2, ny=(1854-ns[cl])*(1853-ns[cl])/2, 
                               eftype='cohen', decimals=3)
    axs[cl].fill_between(x, low1, up1, color = 'red', alpha=alfa)
    
    axs[cl].plot(x, lum_med_cohen[cl], color='blue', label='Median')
    low2,up2 = pg.compute_esci(stat=lum_med_cohen[cl], 
                               nx=(ns[cl]*ns[cl]-1)/2, ny=(1854-ns[cl])*(1853-ns[cl])/2, 
                               eftype='cohen', decimals=3)
    axs[cl].fill_between(x, low2, up2, color = 'blue', alpha=alfa)
    axs[cl].plot(x,np.zeros(275), color='grey')
    
    sig = [False]*50 + clusters_cohen_lda_med['lda'][cl][50:85] + [False]*190
    sig2 = [False]*50 + clusters_cohen_lda_med['med'][cl][50:85] + [False]*190
    
    if cl == 2:
    	linea(axs[cl], sig, y=0.5, color='red')
    	linea(axs[cl], sig2, y=0.55, color='blue')
    	linea(axs[cl], solapamiento(low1, up1, low2, up2), y=0.6, color='black')
    else:
        linea(axs[cl], sig, y=0.6, color='red') 
        linea(axs[cl], sig2, y=0.65, color='blue')
        linea(axs[cl], solapamiento(low1, up1, low2, up2), y=0.7, color='black')
    
    '''SPEARMAN'''
    p = cl+4
    
    corr_lda = [lum_lda_spearman[cl][t][0] for t in range(275)]
    axs[p].plot(x, corr_lda, color='red', label='LDA')
    low,up = pg.compute_esci(stat=corr_lda, nx=1717731, ny=1717731, eftype='r', decimals=3)
    axs[p].fill_between(x, low,up, color='red', alpha=0.2)
    sig = [False]*50 + [lum_lda_spearman[cl][t][1]<0.05 for t in range(50,85)] + [False]*190
    linea(axs[p], sig, 0.06, color='red')
    
    corr_med = [lum_med_spearman[cl][t][0] for t in range(275)]
    axs[p].plot(x, corr_med, color='blue', label='Median')
    low,up = pg.compute_esci(stat=corr_med, nx=1717731, ny=1717731, eftype='r', decimals=3)
    axs[p].fill_between(x, low,up, color='blue', alpha=0.2)
    sig = [False]*50 + [lum_med_spearman[cl][t][1]<0.05 for t in range(50,85)] + [False]*190
    sig2 = [False]*50 + [dependent_corr(lum_med_spearman[cl][t][0],lum_lda_spearman[cl][t][0],
                lda_med_spearman[cl][t][0], n=1717731)[1]<0.05 for t in range(50,85)] + [False]*190
    
    if p in [4,5]:
        linea(axs[p], sig, 0.075, color='blue')
        linea(axs[p], sig2, 0.092, color='blue')
    else:
        linea(axs[p], sig, 0.07, color='blue')
        linea(axs[p], sig2, 0.08, color='black')      
    
    axs[p].plot(x,np.zeros(275), color='grey')

leg = axs[0].legend(loc=[0.5,0.7], fontsize=40, frameon=False, handlelength=1)
for legobj in leg.legend_handles:
    legobj.set_linewidth(4.0)
    
for p in range(8):
    axs[p].set_xticks([0,200,400,600,800])
    if p != 3 and p!= 7:
        axs[p].tick_params(axis='x', width=1.5, labelbottom=False)
        axs[p].tick_params(axis='y', labelsize=27, width=1.5, length=10)
    else:
        axs[p].tick_params(axis='both', width=1.5,  length=10, labelsize=30)
        axs[p].set_xticklabels(["0", "200", "400", "600", "800"], color="k", size=30)
axs[3].set_yticks([0,0.25,0.5])
axs[2].set_yticks([0,0.25,0.5])

# axs[3].set_ylabel("Cohen's d", fontsize=45, labelpad=10)
# axs[7].set_ylabel("Spearman's rho", fontsize=45, labelpad=10, loc='right')
axs[3].text(x=-440,y=1.8,s="Cohen's d", rotation=90, fontsize=54)
axs[7].text(x=-440,y=0.2,s="Spearman's correlation", rotation=90, fontsize=54)
axs[7].text(x=-650,y=-0.18,s="Time (ms)", fontsize=54)

# plt.savefig(carpeta_figuras + 'fig6.png', edgecolor='black', dpi=800, facecolor='white', transparent=True)

#%% Fig 7 Datos
#archivo: cohen_spearman_categorias

f_dic = '/home/ubuntuverde/Eric/rsvp/scripts/clustering/'
path7 = '/home/ubuntuverde/Eric/rsvp/scripts/categorias/'
with open(path7 + 'cat_lda_cohen.pkl', 'rb') as f:
      cohen_lda = pickle.load(f)
with open(path7 + 'cat_med_cohen.pkl', 'rb') as f:
      cohen_med = pickle.load(f)
with open(path7 + 'cat_lda_spearman.pkl', 'rb') as f:
      spearman_lda = pickle.load(f)
with open(path7 + 'cat_med_spearman.pkl', 'rb') as f:
      spearman_med = pickle.load(f)
with open(path7 + 'dic_positivos.pkl', 'rb') as f:
    positivos = pickle.load(f)
with open(f_dic + 'lda_med_spearman.pkl', 'rb') as f:
    lda_med_spearman = pickle.load(f)

#%% Fig 7 Grafico
alfa = 0.1
fig, axs = plt.subplots(2, 4)
plt.subplots_adjust(bottom=0.1, wspace=0.25)
fig.set_size_inches(34, 14)

axs[0,0].set_title('a', x=-0.1, fontsize=45, y=1.05)
axs[0,1].set_title('b', x=-0.1, fontsize=45, y=1.05) 
axs[0,2].set_title('c', x=-0.1, fontsize=45, y=1.05)
axs[0,3].set_title('d', x=-0.1, fontsize=45, y=1.05)
axs[1,0].set_title('e', x=-0.1, fontsize=45, y=1.05)
axs[1,1].set_title('f', x=-0.1, fontsize=45, y=1.05)
axs[1,2].set_title('g', x=-0.1, fontsize=45, y=1.05)
axs[1,3].set_title('h', x=-0.1, fontsize=45, y=1.05)

for i in range(2):
    for j in range(4):
        axs[i,j].spines['left'].set_position(('axes',0))
        axs[i,j].spines['left'].set_color('black')
        
        # turn off the right spine/ticks
        axs[i,j].spines['right'].set_color('none')
        axs[i,j].yaxis.tick_left()
        
        # turn off the top spine/ticks
        axs[i,j].spines['top'].set_color('none')
        axs[i,j].margins(x=0)
        axs[i,j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# fig.suptitle('Fig. 7', x=0.07, y=1.05, size=55)

cats = ['human', 'body', 'animal', 'hardware']
coin = [703, 528, 16110, 11935]
dis = [1717731-c for c in coin]

for c in range(4):
    
    '''COHEN LDA Y MEDIANA'''  
    axs[0,c].plot(x, cohen_lda[cats[c]], color='red', label='LDA', alpha=1)
    low1,up1 = pg.compute_esci(stat=np.array(cohen_lda[cats[c]]), nx=coin[c], ny=dis[c], 
                               eftype='cohen', decimals=3)
    axs[0,c].fill_between(x, low1, up1, color = 'red', alpha=alfa)
    
    axs[0,c].plot(x, cohen_med[cats[c]], color='blue', label='Median', alpha=1)
    low2,up2 = pg.compute_esci(stat=np.array(cohen_med[cats[c]]), nx=coin[c], ny=dis[c], 
                               eftype='cohen', decimals=3)
    axs[0,c].fill_between(x, low2, up2, color = 'blue', alpha=alfa)
    
    axs[0,c].tick_params(axis='x', which='both', labelbottom= False)
    axs[0,c].tick_params(axis='y', which='both', labelsize=25)
    axs[0,c].plot(x,np.zeros(275),color='grey')
    axs[0,c].locator_params(axis='y', nbins=5)
    axs[0,c].tick_params(axis='x', width=1.5, labelbottom=False)
    axs[0,c].tick_params(axis='y', labelsize=27, width=1.5, length=10)
    sig = [False]*50 + positivos['lda'][cats[c]][50:85] + [False]*190
    linea(axs[0,c], sig, y=0.5, color='red')
    sig2 = [False]*50 + positivos['med'][cats[c]][50:85] + [False]*190
    linea(axs[0,c], sig2, y=0.52, color='blue')
    linea(axs[0,c], solapamiento(low1,up1,low2,up2), y=0.54, color='black')
    
    '''SPEARMAN LDA Y MEDIANA'''
    axs[1,c].set_xticks([0, 200,400,600,800])
    axs[1,c].tick_params(axis='both', width=1.5,  length=10, labelsize=27)
    axs[1,c].plot(x, [spearman_lda[cats[c]][t][0] for t in range(275)], color='red', label='LDA')
    axs[1,c].plot(x, [spearman_med[cats[c]][t][0] for t in range(275)], color='blue', label='Median')
    axs[1,c].tick_params(axis='both', which='major', labelsize=27)
    axs[1,c].plot(x,np.zeros(275),color='grey')
    sig = [False]*50 + [spearman_lda[cats[c]][t][1]<0.05 for t in range(50,85)] + [False]*190 
    if c == 0 or c == 1:
        linea(axs[1,c], sig, 0.019, color='red')
    else:
        linea(axs[1,c], sig, 0.02, color='red')
    sig = [False]*50 + [spearman_med[cats[c]][t][1]<0.05 for t in range(50,85)] + [False]*190 
    if c == 0 or c == 1:
        linea(axs[1,c], sig, 0.020, color='blue')
    else:
        linea(axs[1,c], sig, 0.022, color='blue')
    sig = [False]*50 + [dependent_corr(spearman_lda[cats[c]][t][0],spearman_med[cats[c]][t][0], 
                         lda_med_spearman[c][t][0], n=1717731)[1]<0.05 for t in range(50,85)] + [False]*190
    if c == 0 or c == 1:
        linea(axs[1,c], sig, 0.021, color='black')
    else:
        linea(axs[1,c], sig, 0.024, color='black')
    
axs[1,1].text(x=550,y=-0.025,s="Time (ms)", fontsize=55) 
leg = axs[1,3].legend(loc=[0.4,2.1], fontsize=45, frameon=False, handlelength=0.5)
for legobj in leg.legend_handles:
    legobj.set_linewidth(4.0)
axs[0,0].set_ylabel("Cohen's d", fontsize=45, labelpad=33)
axs[1,0].set_ylabel("Spearman's' correlation", fontsize=45, labelpad=33)

# plt.savefig(carpeta_figuras + 'fig7.png', edgecolor='black', dpi=600, facecolor='white', transparent=True)

#%% Fig 8 Datos
path8 = '/home/ubuntuverde/Eric/rsvp/scripts/categorias/reemplazos/'

with open(path8 + 'dic_med.pkl', 'rb') as f:
    gran_med = pickle.load(f)
                          
with open(path8 + 'dic_lda.pkl', 'rb') as f:
    gran_lda = pickle.load(f)
        
cats = ['human', 'body', 'animal', 'hardware']

#%% Fig 8 Grafico

fig, axs = plt.subplots(2, 4)
plt.subplots_adjust(bottom=0.1, wspace=0.3)
fig.set_size_inches(32, 15)

axs[0,0].set_title('a', x=-0.1, fontsize=45, y=1.05)
axs[0,1].set_title('b', x=-0.1, fontsize=45, y=1.05) 
axs[0,2].set_title('c', x=-0.1, fontsize=45, y=1.05)
axs[0,3].set_title('d', x=-0.1, fontsize=45, y=1.05)
axs[1,0].set_title('e', x=-0.1, fontsize=45, y=1.05)
axs[1,1].set_title('f', x=-0.1, fontsize=45, y=1.05)
axs[1,2].set_title('g', x=-0.1, fontsize=45, y=1.05)
axs[1,3].set_title('h', x=-0.1, fontsize=45, y=1.05)

for i in range(2):
    for j in range(4):
        axs[i,j].spines['left'].set_position(('axes',0))
        axs[i,j].spines['left'].set_color('black')
        
        # turn off the right spine/ticks
        axs[i,j].spines['right'].set_color('none')
        axs[i,j].yaxis.tick_left()
        
        # set the y-spine
        axs[i,j].spines['bottom'].set_position(('axes',0))
        axs[i,j].spines['bottom'].set_color('black')
        
        # turn off the top spine/ticks
        axs[i,j].spines['top'].set_color('none')
        axs[i,j].xaxis.tick_bottom()
        axs[i,j].margins(x=0)
        axs[i,j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[i,j].tick_params(axis='both', width=1.5,  length=10, labelsize=27)
        axs[i,j].set_xticks([0, 25, 50, 75, 100])

fig.suptitle('Fig. 8', x=0.06, y=1.0, size=55)

for c in range(4):
    xi = np.linspace(0,100*len(gran_lda[cats[c]][0])/len(cat_con[cats[c]]),len(gran_lda[cats[c]][0]))
    
    '''LDA'''
    for r in range(len(gran_lda[cats[c]])):
        y = gran_lda[cats[c]][r]
        axs[0,c].scatter(xi,y, s=5, color='green', alpha=0.5)
        m, b = np.polyfit(xi,y,1)
        axs[0,c].plot(xi, (m*xi)+b, color = 'red', alpha=0.2)
    if c == 2:  #solo para que el eje X llegue hasta el 100
        axs[0,2].plot(100,0.16,color='white')

    prom = np.mean(gran_lda['estadisticos'][cats[c]]['pendientes'])
    promb = np.mean(gran_lda['estadisticos'][cats[c]]['ordenadas'])
    pval = gran_lda['estadisticos'][cats[c]]['pvalor']
    
    axs[0,c].plot(xi, prom*xi+promb, color='red', linewidth=4)
    axs[0,c].tick_params(axis='x', which='both', labelbottom= False)
    axs[0,c].tick_params(axis='y', which='both', labelsize=25)
    axs[0,c].locator_params(axis='y', nbins=5)


    '''Mediana'''
    for r in range(len(gran_med[cats[c]])):
        y = gran_med[cats[c]][r]
        axs[1,c].scatter(xi,y, s=5, color='green', alpha=0.5)
        m, b = np.polyfit(xi,y,1)
        axs[1,c].plot(xi, (m*xi)+b, color = 'blue', alpha=0.2)
    if c == 2:  #solo para que el eje X llegue hasta el 100
        axs[1,2].plot(100,0.31,color='white')

    prom = np.mean(gran_med['estadisticos'][cats[c]]['pendientes'])
    promb = np.mean(gran_med['estadisticos'][cats[c]]['ordenadas'])
    pval = gran_med['estadisticos'][cats[c]]['pvalor']

    axs[1,c].plot(xi, prom*xi+promb, color='blue', linewidth=4)
    axs[1,c].tick_params(axis='y', which='both', labelsize=35)
    axs[1,c].tick_params(axis='both', which='major', labelsize=27)
    axs[1,c].locator_params(axis='y', nbins=5)

axs[1,1].text(x=50,y=0.15,s="Degradation (%)", fontsize=55) 
axs[1,1].text(x=-175,y=0.19,s='Non coincidental - coincidental', fontsize=55, rotation=90)

frase1 = 'Human: pendientes = {}, {}; pval = {}, {}.'.format(np.mean(gran_lda['estadisticos']['human']['pendientes']),
                                                            np.mean(gran_med['estadisticos']['human']['pendientes']),
                                                            gran_lda['estadisticos']['human']['pvalor'],
                                                            gran_med['estadisticos']['human']['pvalor'])
frase2 = 'Body: pendientes = {}, {}; pval = {}, {}.'.format(np.mean(gran_lda['estadisticos']['body']['pendientes']),
                                                            np.mean(gran_med['estadisticos']['body']['pendientes']),
                                                            gran_lda['estadisticos']['body']['pvalor'],
                                                            gran_med['estadisticos']['body']['pvalor'])
frase3 = 'Animal: pendientes = {}, {}; pval = {}, {}.'.format(np.mean(gran_lda['estadisticos']['animal']['pendientes']),
                                                            np.mean(gran_med['estadisticos']['animal']['pendientes']),
                                                            gran_lda['estadisticos']['animal']['pvalor'],
                                                            gran_med['estadisticos']['animal']['pvalor'])
frase4 = 'Hardware: pendientes = {}, {}; pval = {}, {}.'.format(np.mean(gran_lda['estadisticos']['hardware']['pendientes']),
                                                            np.mean(gran_med['estadisticos']['hardware']['pendientes']),
                                                            gran_lda['estadisticos']['hardware']['pvalor'],
                                                            gran_med['estadisticos']['hardware']['pvalor'])

# print(frase1, frase2, frase3)
# print(frase4)

# plt.savefig(carpeta_figuras + 'fig8.png', edgecolor='black', dpi=1200, facecolor='white', transparent=True)

