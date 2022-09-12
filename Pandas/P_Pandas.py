#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[5]:


#crear Data frame
datos = {'Mes':['Enero', 'Febrero', 'Marzo', 'Abril'], 'Ventas':[30500, 35600, 28300, 33900], 'Gastos':[22000, 23400, 18100, 20700]}
contabilidad = pd.DataFrame(datos)
print(contabilidad)


# In[8]:


#Balance (ventas - gastos) Data frame anterior
def balance(contabilidad, meses):
    contabilidad['Balance'] = contabilidad.Ventas - contabilidad.Gastos
    return contabilidad[contabilidad.Mes.isin(meses)].Balance.sum()
print(balance(contabilidad, ['Enero','Febrero','Marzo','Abril']))


# In[15]:


#Importar Dara Frame
contaminacion = pd.read_csv("contaminacion.csv") 
print(contaminacion)


# In[20]:


#Matriz de correlación de variables
contaminacion.corr("pearson")
print(contaminacion.corr("pearson"))

# In[21]:


#Estadisticos descrptivos del dataframe
contaminacion.describe()
print(contaminacion.describe())

# In[22]:


#Trasponer filas y columnas del Data Frame
contaminacion.T
print(contaminacion.T)

# In[24]:


#Extraer columnas del Data Frame
contaminacion[['Contaminacion_SO2','Habitantes','Lluvia']]
print(contaminacion[['Contaminacion_SO2','Habitantes','Lluvia']])

# In[26]:


#Aplicar filtros en el Data Frame
contaminacion[contaminacion['Contaminacion_SO2']>30]
print(contaminacion[contaminacion['Contaminacion_SO2']>30])

# In[ ]:




