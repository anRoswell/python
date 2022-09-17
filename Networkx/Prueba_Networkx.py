#!/usr/bin/env python
# coding: utf-8

# In[2]:


import networkx as nx


# In[7]:


G = nx.Graph() # crear un grafo
#Añadir nodos
G.add_node("Kevin Bacon")
G.add_node("Tom Hanks")
G.add_nodes_from(["Meg Ryan", "Parker Posey", "Lisa Kudrow"])
#Añadir aristas
G.add_edge("Kevin Bacon", "Tom Hanks")
G.add_edge("Kevin Bacon", "Meg Ryan")
G.add_edges_from([("Tom Hanks", "Meg Ryan"), ("Tom Hanks", "Parker Posey")])
G.add_edges_from([("Parker Posey", "Meg Ryan"), ("Parker Posey", "Lisa Kudrow")])
print(len(G.nodes))
print(len(G.edges))
print(G.nodes)
print(G.edges)


# In[8]:


# asignar atributos a nodos y aristas
G.nodes["Tom Hanks"]["oscars"] = 2
G.edges["Kevin Bacon", "Tom Hanks"]["pelicula"] = "Apolo 13"
G.edges["Kevin Bacon", "Meg Ryan"]["pelicula"] = "En carne viva"
G.edges["Parker Posey", "Meg Ryan"]["pelicula"] = "Algo para recordar"
G.edges["Parker Posey", "Tom Hanks"]["pelicula"] = "Tienes un email"
G.edges["Parker Posey", "Lisa Kudrow"]["pelicula"] = "Esperando la hora"
print(G.nodes(data=True))
print(G.edges(data=True))
print(G["Kevin Bacon"])


# In[23]:


import networkx as nx

G = nx.DiGraph()
G.add_edges_from(
    [('A', 'B'), ('A', 'C'), ('D', 'B'), ('E', 'C'), ('E', 'F'),
     ('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G')])

val_map = {'A': 1.0,
           'D': 0.5714285714285714,
           'H': 0.0}

values = [val_map.get(node, 0.25) for node in G.nodes()]

# Specify the edges you want here
red_edges = [('A', 'C'), ('E', 'C')]
edge_colours = ['black' if not edge in red_edges else 'red'
                for edge in G.edges()]
black_edges = [edge for edge in G.edges() if edge not in red_edges]

print(black_edges, red_edges)


# In[24]:


import pandas as pd
import numpy as np
import networkx as nx
import openpyxl as op

# In[44]:


#Algoritmo de Dijkstra para determinar ruta mas corta
df = pd.read_csv("estaciones.csv")
print(df)
ESTACION = nx.from_pandas_edgelist(df,source='start',target='end',edge_attr='distancia')
djk_path= nx.dijkstra_path(ESTACION, source='E1', target='E9', weight='distancia')
print(djk_path)
len(djk_path)

