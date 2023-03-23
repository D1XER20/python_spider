import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
#zadanie 1.2a
data = pd.read_excel('practice_lab_1.xlsx')

tablica = np.array(data)

kolumny = list(data.columns)

tab1 = tablica[0::2,:]
tab2 = tablica[1::2,:]
wynik = tab1-tab2

print(wynik)
#%%
#zadanie1.2b
tab_odch = tablica.std()
tab_sr = tablica.mean()

tab =(tablica - tab_sr)/tab_odch

#%%
#zadanie1.2c
tab_odch = tablica.std(axis=0)
tab_sred = tablica.mean(axis=0)

tab =(tablica - tab_sr)/(tab_odch + np.spacing(tablica.std(axis=0)))

#%%
#zad1.2d
tab_stosunek = tab_sred/(tab_odch + np.spacing(tablica.std(axis=0)))

#%%
#zad1.2e
np.argmax(tab_stosunek)

#%%
#zad1.2f
arr6 = (tablica>tablica.mean(axis = 0)).sum(axis=0)
#%%
#%%
#zad1.2g
arr_7 = (tablica[:,::1] > tablica.mean(axis=0)).sum(axis=0)  
#%%
#%%
#zad1.2j
mask = (tablica == tablica.max())[:,::1].sum(axis=0)>0
arr_8 = np.array(kolumny)[mask]
#%%
#zad1.3
#5
x = np.arange(-5,5,0.01)
plt.plot(x[x>0],x[x>0])
plt.plot(x[x<=0],np.exp(x[x<=0])-1)
#4
plt.plot(x[x>0],x[x>0])
plt.plot(x[x<=0],np.zeros(len(x[x<=0])))
         
#2
y = (np.exp(x) - np.exp(-x))/(np.exp(x)+np.exp(-x))
plt.plot(x,y)
#1
y=(np.tanh(x))
plt.plot(x,y)
#3
y=(1/(1+np.exp(-x)))
plt.plot(x,y)