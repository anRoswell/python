#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import scipy as sp


# In[8]:


from scipy.optimize import fminbound


# In[34]:


import matplotlib.pyplot as plt 


# In[40]:


#Calculo del minimo en un iontervalo de la función y=-cos((a*pi*x)/b)+cx^d
def mi_funcion(x, a, b, c, d):
    y = -np.cos(a*sp.pi*x/b) + c*x**d
    return y
a = 2
 
b = 0.5
 
c = 0.05
 
d = 2

x1 = 0.2
 
x2 = 0.6
 
xt=np.arange(0,1,.01)
 
yt = -np.cos(a*sp.pi*xt/b) + c*xt**d

x_minimo = fminbound(mi_funcion,x1,x2, args = (a,b,c,d))
 
ysol = mi_funcion(x_minimo, a, b, c, d)
print (u'El minimo esta en x = %2.3f, y = %2.3f' %(x_minimo, ysol))


# In[41]:


from scipy.optimize import curve_fit # Importamos curve_fit de scipy.optimize


# In[44]:


# Ajuste de una función: ventas= e^(-bx^2/2d^2)+cx

def mi_funcion(x, a, b, c, d):
 
  return a*np.exp(-b*x**2/(2*d**2)) + c * x

x = np.linspace(0, 5,40)
 
y = mi_funcion(x, 2.5, 1.3, 0.5,1)

def ruido(x,y,k):
 
  yn = y + k * sp.random.normal(size = len(x))
 
  return yn

coeficientes_optimizados, covarianza_estimada = curve_fit(mi_funcion, x, y)
print ("Coeficientes optimizados:", coeficientes_optimizados)
 
print ("Covarianza estimada:", covarianza_estimada)


# In[50]:


from scipy import integrate


# In[57]:


#calculo de integrales: y=e^-x
def integral_1(limite_inferior, limite_superior, mostrar_resultados):
  # funcion e^(-x)
  exponencial_decreciente = lambda x: np.exp(-x)
  # resultados por pantalla
  if mostrar_resultados == True:
    print ('La integral entre %2.2f y %2.2f es '% (limite_inferior, limite_superior))
    print(integrate.quad(exponencial_decreciente,limite_inferior,limite_superior))
  # Los devuelvo
  return integrate.quad(exponencial_decreciente ,limite_inferior,limite_superior)
 
print(integral_1(limite_inferior = 0, limite_superior = sp.inf, mostrar_resultados = True))


# In[53]:


from scipy import interpolate


# In[60]:


#interpolación de datos
#array
 
x = np.linspace(0,3,10)
 
# generamos datos experimentales de ejemplo)
 
y = np.exp(-x/3.0)
 
# Interpol
 
interpolacion = interpolate.interp1d(x, y)
 
# array con mas puntos en el mismo intervalo
 
x2 = np.linspace(0,3,1000)
 
# Evaluamos x2 en la interpolacion
 
y2 = interpolacion(x2)

print(x2,y2)


# In[63]:


#Calculo de las raíces en un polinomio
# Creamos un polinomio
polinomio = [4.3,9,.6,-1]# polinomio = 4.3 x^3 + 9 x^2 + 0.6 x - 1
# array
x = np.arange(-4,2,.05)
#&amp;amp;nbsp; Evaluamos el polinomio en x mediante polyval.
y = np.polyval(polinomio,x)
# Calculamos las raices del polinomio 
raices = np.roots(polinomio)
# Evaluamos el polinomio en las raices
s = np.polyval(polinomio,raices)
# Las presentamos en pantalla
print ("Las raices son %2.2f, %2.2f, %2.2f. " % (raices[0], raices[1], raices[2]))


# In[ ]:




