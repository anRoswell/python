#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 


# In[2]:


#Crear array apartir de listas
numeros_primos = [2,3,5,7,11,13,17,19,23,29]
array_primos = np.array(numeros_primos)
print(array_primos)


# In[3]:


#Crear array con ceros 
array_zero = np.zeros(10)
print(array_zero)


# In[4]:


#Crear array con numeros 
array_numeros = np.arange(15)
print(array_numeros)


# In[5]:


#Array con numeros sucesivos 
array_pares = np.arange(0,20,2)
print(array_pares)


# In[12]:


#Ordenar un no array
numeros_fibonacci = [55,0,144,1,21,89,5,8,13,1,34,3,2]
array_fibonacci=np.array(numeros_fibonacci)
print(np.sort(array_fibonacci))


# In[20]:


#Crear Matriz
a = np.arange(15).reshape(3,5)
print(a)


# In[16]:


#Crear matriz de identidad
print(np.eye(5))


# In[18]:


#Crear numeros aleatorios
print(np.random.randn(30))


# In[ ]:




