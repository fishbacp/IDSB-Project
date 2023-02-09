#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 10:17:18 2019

@author: fishbacp
"""

'''
XKCD_COLORS = {
    'cloudy blue': '#acc2d9',
    'dark pastel green': '#56ae57',
    'dust': '#b2996e',
    'electric lime': '#a8ff04',
    'fresh green': '#69d84f',
    'light eggplant': '#894585',
    'nasty green': '#70b23f',
    'really light blue': '#d4ffff',
    'tea': '#65ab7c',
    'warm purple': '#952e8f',
    'yellowish tan': '#fcfc81',
    'cement': '#a5a391',
    'dark grass green': '#388004',
    'dusty teal': '#4c9085',
    'grey teal': '#5e9b8a',
    'macaroni and cheese': '#efb435',
    'pinkish tan': '#d99b82',
    'spruce': '#0a5f38',
    'strong blue': '#0c06f7',
    'toxic green': '#61de2a',
    'windows blue': '#3778bf',
    'blue blue': '#2242c7',
    'blue with a hint of purple': '#533cc6',
    'booger': '#9bb53c',
    'bright sea green': '#05ffa6',
    'dark green blue': '#1f6357',
    'deep turquoise': '#017374',
    'green teal': '#0cb577',
    'strong pink': '#ff0789',
    'bland': '#afa88b',
    'deep aqua': '#08787f',
    'lavender pink': '#dd85d7',
    'light moss green': '#a6c875',
    'light seafoam green': '#a7ffb5',
    'olive yellow': '#c2b709',
    'pig pink': '#e78ea5',
    'deep lilac': '#966ebd',
    'desert': '#ccad60',
    'dusty lavender': '#ac86a8',
    'purpley grey': '#947e94',
    'purply': '#983fb2',
    'candy pink': '#ff63e9',
    'light pastel green': '#b2fba5',
    'boring green': '#63b365',
    'kiwi green': '#8ee53f',
    'light grey green': '#b7e1a1',
    'orange pink': '#ff6f52',
    'tea green': '#bdf8a3',
    'very light brown': '#d3b683',
    'egg shell': '#fffcc4',
    'eggplant purple': '#430541',
    'powder pink': '#ffb2d0',
    'reddish grey': '#997570',
    'baby shit brown': '#ad900d',
    'liliac': '#c48efd',
    'stormy blue': '#507b9c',
    'ugly brown': '#7d7103',
    'custard': '#fffd78',
    'darkish pink': '#da467d',
    'deep brown': '#410200',
    'greenish beige': '#c9d179',
    'manilla': '#fffa86',
    'off blue': '#5684ae',
    'battleship grey': '#6b7c85',
    'browny green': '#6f6c0a',
    'bruise': '#7e4071',
    'kelley green': '#009337',
    'sickly yellow': '#d0e429',
    'sunny yellow': '#fff917',
    'azul': '#1d5dec',
    'darkgreen': '#054907',
    'green/yellow': '#b5ce08',
    'lichen': '#8fb67b',
    'light light green': '#c8ffb0',
    'pale gold': '#fdde6c',
    'sun yellow': '#ffdf22',
    'tan green': '#a9be70',
    'burple': '#6832e3',
    'butterscotch': '#fdb147',
    'toupe': '#c7ac7d',
    'dark cream': '#fff39a',
    'indian red': '#850e04',
    'light lavendar': '#efc0fe',
    'poison green': '#40fd14',
    'baby puke green': '#b6c406',
    'bright yellow green': '#9dff00',
    'charcoal grey': '#3c4142',
    'squash': '#f2ab15',
    'cinnamon': '#ac4f06',
    'light pea green': '#c4fe82',
    'radioactive green': '#2cfa1f',
    'raw sienna': '#9a6200',
    'baby purple': '#ca9bf7',
    'cocoa': '#875f42',
    'light royal blue': '#3a2efe',
    'orangeish': '#fd8d49',
    'rust brown': '#8b3103',
    'sand brown': '#cba560',
    'swamp': '#698339',
    'tealish green': '#0cdc73',
    'burnt siena': '#b75203',
    'camo': '#7f8f4e',
    'dusk blue': '#26538d',
    'fern': '#63a950',
    'old rose': '#c87f89',
    'pale light green': '#b1fc99',
    'peachy pink': '#ff9a8a',
    'rosy pink': '#f6688e',
    'light bluish green': '#76fda8',
    'light bright green': '#53fe5c',
    'light neon green': '#4efd54',
    'light seafoam': '#a0febf',
    'tiffany blue': '#7bf2da',
    'washed out green': '#bcf5a6',
    'browny orange': '#ca6b02',
    'nice blue': '#107ab0',
    'sapphire': '#2138ab',
    'greyish teal': '#719f91',
    'orangey yellow': '#fdb915',
    'parchment': '#fefcaf',
    'straw': '#fcf679',
    'very dark brown': '#1d0200',
    'terracota': '#cb6843',
    'ugly blue': '#31668a',
    'clear blue': '#247afd',
    'creme': '#ffffb6',
    'foam green': '#90fda9',
    'grey/green': '#86a17d',
    'light gold': '#fddc5c',
    'seafoam blue': '#78d1b6',
    'topaz': '#13bbaf',
    'violet pink': '#fb5ffc',
    'wintergreen': '#20f986',
    'yellow tan': '#ffe36e',
    'dark fuchsia': '#9d0759',
    'indigo blue': '#3a18b1',
    'light yellowish green': '#c2ff89',
    'pale magenta': '#d767ad',
    'rich purple': '#720058',
    'sunflower yellow': '#ffda03',
    'green/blue': '#01c08d',
    'leather': '#ac7434',
    'racing green': '#014600',
    'vivid purple': '#9900fa',
    'dark royal blue': '#02066f',
    'hazel': '#8e7618',
    'muted pink': '#d1768f',
    'booger green': '#96b403',
    'canary': '#fdff63',
    'cool grey': '#95a3a6',
    'dark taupe': '#7f684e',
    'darkish purple': '#751973',
    'true green': '#089404',
    'coral pink': '#ff6163',
    'dark sage': '#598556',
    'dark slate blue': '#214761',
    'flat blue': '#3c73a8',
    'mushroom': '#ba9e88',
    'rich blue': '#021bf9',
    'dirty purple': '#734a65',
    'greenblue': '#23c48b',
    'icky green': '#8fae22',
    'light khaki': '#e6f2a2',
    'warm blue': '#4b57db',
    'dark hot pink': '#d90166',
    'deep sea blue': '#015482',
    'carmine': '#9d0216',
    'dark yellow green': '#728f02',
    'pale peach': '#ffe5ad',
    'plum purple': '#4e0550',
    'golden rod': '#f9bc08',
    'neon red': '#ff073a',
    'old pink': '#c77986',
    'very pale blue': '#d6fffe',
    'blood orange': '#fe4b03',
    'grapefruit': '#fd5956',
    'sand yellow': '#fce166'}
color_names=list(XKCD_COLORS.values())

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy.linalg as LA
import scipy as sc

import networkx as nx
import community
import leidenalg as la
import igraph as ig

import networkx.algorithms.community as nc

from networkx.algorithms.community.quality import modularity
from networkx.algorithms.smallworld import omega, random_reference, lattice_reference


import skfuzzy as fuzzy # listed under scikit-fuzzy at pypl
from sklearn.cluster import SpectralClustering


"""
network_draw: Create and plot undirected graph from a symmetric adjacency matrix, A.

Arguments: A (symmetric adjacency matrix)
pos (e.g.,pos = nx.spring_layout(G))
labels_dict: (a dictionary of labels, e.g. {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'})
other options to determine node color node, etc

Returns: Graph G
"""

def network_draw(A, labels_dict, node_color='red',node_size=1000,font_size=17,edge_font=8,weighted=False,file_name='graph_undirected.png'):
    B=np.around(A,2)
    G=nx.from_numpy_matrix(B)
    pos=nx.spring_layout(G)
    nx.draw_networkx_nodes(G,pos,
                       nodelist=[j for j in range(0,len(A))],
                       node_color=node_color,
                       node_size=node_size,ax=None,
                   alpha=0.8)
    nx.draw_networkx_labels(G,pos,labels_dict,font_size=font_size)
    nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
    if weighted:
        #edge_labels=dict([((u,v,),np.around(d['weight'],2))
        edge_labels = dict([((u,v,), f"{d['weight']:.2f}") for u,v,d in G.edges(data=True)])
        #for u,v,d in G.edges(data=True)])
        nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,font_size=edge_font)
    ax=plt
    ax.axis('off')
    fig = ax.gcf()
    plt.show()
    fig.savefig(file_name)
    return G

"""
network_draw: Create and plot a weighted, directed graph from a adjacency matrix, A.

Arguments: A (adjacency matrix)
labels_dict: (a dictionary of labels, e.g. {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'})
file_name: (a file name for the image, whicch is saved but can be viewed afterwards),
e.g. file_name='directed.png'
other options to determine node color node, etc

Returns: Graph G


To view after invoking command use:
    from IPython.display import Image
    Image(file_name)
"""

def network_directed(A,labels_dict,file_name='directed.png',node_size=1000,node_color='blue',node_shape='circle',fill_color='yellow',node_font=14,show_weights='False',edge_font=14,edge_color='black',arrow_size=1):
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    nx.set_node_attributes(G, {k: {'label': labels_dict[k]} for k in labels_dict.keys()})
    G.graph['edge']={'splines':'curved'}
    G.graph['edge'] = {'arrowsize': '0.8', 'splines':'curved','shape':'circle','color':'magenta'}
    #nx.set_node_attributes(G,node_size=node_size,node_font=node_font)
    #G.graph['node_size']=node_size
    if show_weights:
        nx.set_edge_attributes(G, {(e[0], e[1]): {'label': e[2]['weight']} for e in G.edges(data=True)})
    D = nx.drawing.nx_agraph.to_agraph(G)
    # Modify node fillcolor and edge color.
    D.node_attr.update(color=node_color, style='filled', shape=node_shape,fillcolor=fill_color,size=node_size,fontsize=node_font)
    D.edge_attr.update(color=edge_color,arrowsize=arrow_size,splines='curved',fontsize=edge_font)
    pos = D.layout('dot')
    D.draw(file_name)
    #ax=plt
    #ax.axis('off')
    #fig = ax.gcf()
    #plt.show()
    #fig.savefig(file_name)
    # Return original directed graph for future calculations.
    return G

"""
network_from_edgelist: Import text file containing graph in edge list format.

Arguments: filename (textfile containing graph in edge list format)

# Returns: Graph G (undirected or directed) and adjacency matrix A
"""

def network_from_edgelist(filename):
    edge_list=np.loadtxt(filename)
    dim=int(max(edge_list[:,0:1])[0])
    A=np.zeros((dim,dim))
    if edge_list.shape[1]==2:
        for row in edge_list:
            A[int(row[0]-1),int(row[1]-1)]=1
    else:
        for row in edge_list:
            A[int(row[0]-1),int(row[1]-1)]=row[2]
    def check_symmetric(A, rtol=1e-05, atol=1e-08):
        return np.allclose(A, A.T, rtol=rtol, atol=atol)
    if check_symmetric(A):
        G=nx.from_numpy_matrix(A)
    else:
        G=nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    return A,G

"""
Example usage:
graph_from_edgelist('EdgeListSample.txt')
"""

"""
network_betweenness_bar: compute betweenness centralities and plot values as a bar graph

Arguments: G (undirected, weighted graph)
labels_dict(a dictionary of labels, e.g. {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'})

Returns: list of betweenness values and plots bar graph,
"""
def network_betweenness_bar(G, labels):
    #Compute Betweenness Centrality Values and plot as bar graph
    BC=nx.betweenness_centrality(G,normalized=False,weight='weight')
    BC=list(BC.values())
    BC=[ round(elem, 2) for elem in BC ]
    x=np.arange(len(labels))
    axis_labels=list(labels.values());
    plt.bar(x,BC)
    plt.xticks(x,axis_labels)
    plt.title('Betweenness Centrality',fontsize=18)
    return BC

"""
network_betweenness: compute betweenness centralities and plot graph with
nodes colored by betweenness value

Arguments: G (undirected, weighted graph)
labels(a dictionary of labels, e.g. {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'})
other options to determine node color node, etc

Returns: list of betweenness values
"""

def network_betweenness(G, labels_dict):
    H=nx.relabel_nodes(G, labels_dict)
    pos=nx.spring_layout(H)
    BC=nx.betweenness_centrality(G,normalized=False,weight='weight')
    BC=list(BC.values())
    BC=[ round(elem, 2) for elem in BC ]
    colors=range(len(BC))
    cmap=plt.cm.cool
    vmin = min(colors)
    vmax = max(colors)
    node_size=500
    font_size=.75*np.sqrt(node_size)
#nx.draw(H, pos, cmap=plt.get_cmap('cool'),node_color=BC, node_size=node_size,font_size=font_size,with_labels=True, vmin=vmin, vmax=vmax)
    edge_labels=dict([((u,v,),d['weight']) for u,v,d in H.edges(data=True)])
    plt.figure()
    nx.draw(H, pos, cmap=plt.get_cmap('cool'),node_color=BC, node_size=node_size,font_size=font_size,with_labels=True, vmin=vmin, vmax=vmax)
    nx.draw_networkx_edge_labels(H, pos,edge_labels=edge_labels)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    sm._A = []
    plt.colorbar(sm)
    plt.title('Betweenness Centrality',fontsize=18)
    plt.show()
    return BC

"""
edge_betweenness: compute eedge betweenness centralities and plot graph with
edges colored by betweenness value

Arguments: G (undirected, weighted graph)
labels_dict(a dictionary of labels, e.g. {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'})
other options to determine node color node, etc

Returns: list of betweenness values
"""

def edge_betweenness(G, labels_dict,node_color='red',node_size=1000,font_size=17,weighted='True'):
    EB=nx.edge_betweenness_centrality(G)
    H=nx.relabel_nodes(G, labels_dict)
    pos=nx.spring_layout(H)
    colors=range(len(EB.values()))
    cmap=plt.cm.cool
    vmin = min(colors)
    vmax = max(colors)
    nx.draw(H, pos,node_color='lightgray', node_size=node_size,font_size=font_size,with_labels=True, edge_color=colors,width=4,edge_cmap=cmap, vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    sm._A = []
    plt.colorbar(sm)
    plt.title('Edge Betweenness',fontsize=18)
    plt.show()
    return EB

"""
network_eigenvector_centrality: compute eigenvectors centralities and plot graph with
nodes colored by betweenness value

Arguments: G (directed, weighted graph, created using nx.DiGraph)
labels_dict (a dictionary of labels, e.g. {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'})
other options to determine node color node, etc

Returns: list of eigenvector centralities
"""

def eigenvector_centrality(G, labels_dict,node_color='red',node_size=1000,font_size=17):
    EC=nx.eigenvector_centrality(G)
    EC=list(EC.values())
    pos = nx.spring_layout(G)
    H=nx.relabel_nodes(G, labels_dict)
    nx.draw(H, cmap=plt.get_cmap('cool'), node_color=EC, with_labels=True, font_color='black')
    fig, ax = plt.subplots(figsize=(8,1))
    fig.subplots_adjust(bottom=0.5)
    cmap = mpl.cm.cool
    norm = mpl.colors.Normalize(vmin=min(EC), vmax=max(EC))
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,norm=norm,orientation='horizontal')
    cb1.set_label('Eigenvectors Centralities',fontsize=18)
    fig.show()
    return EC

def eigenvector_centrality_bar(G, labels_dict):
    #Compute Betweenness Centrality Values and plot as bar graph
    EC=nx.eigenvector_centrality(G,weight='weight')
    EC=list(EC.values())
    EC=[ round(elem, 2) for elem in EC ]
    x=np.arange(len(labels_dict))
    axis_labels=list(labels_dict.values());
    plt.bar(x,EC)
    plt.xticks(x,axis_labels)
    plt.title('Eigenvector Centrality',fontsize=18)
    return EC

"""
louvain: community assignments that maximize modularity based upon Louvain method
Requires python-louvain package available at
https://pypi.org/project/python-louvain/ and described at
https://python-louvain.readthedocs.io/en/latest/
Plots nodes based upon community assignments

Arguments: G (undirected, weighted graph)
labels_dict (a dictionary of labels, e.g. {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'})
other options to determine node color node, etc

# Returns: Q, maximum modularity and community_labels, the community assignments.
"""

def louvain_clusters(S,channels):
    n=len(channels)
    G=nx.from_numpy_matrix(S)
    clusters=nc.louvain_communities(G)
    colors=color_names[0:len(clusters)]
    d = {k:v for s, v in zip(clusters, colors) for k in s}
    color_list = [d[k] for k in sorted(d)]
    return clusters,color_list






def louvain(G,labels_dict,node_size=1000,font_size=17,graph=True):
    partition = community.best_partition(G,weight='weight')

    community_numbers=set(partition.values())
    community_assignments=np.array(list(partition.values()))

    community_assignments=[[y for y in np.where(community_assignments==x)[0]] for x in community_numbers]
    community_labels=[[labels_dict[member] for member in set(comm)] for comm in community_assignments ]
    Q=community.modularity(partition, G)

    #labels=list(labels_dict.values())
    if graph:
        pos = nx.spring_layout(G)
        for comm in community_assignments:
            comm_node_list=comm
            color_index=community_assignments.index(comm)

            #nx.draw_networkx(G,labels=labels_dict,node_color=color_names[color_index])

            nx.draw_networkx_nodes(G, pos, comm_node_list,node_size = node_size,node_color=color_names[color_index])
            nx.draw_networkx_edges(G, pos, alpha=0.5)
            nx.draw_networkx_labels(G,pos,labels_dict,font_size=font_size)
            ax=plt
            ax.axis('off')
            #fig = ax.gcf()
            plt.title('Louvain Clustering',fontsize=18)
        plt.show()

    return community_assignments,Q,community_labels

"""
leiden: community assignments that maximize modularity based upon Leiden method

Arguments: G (directed, weighted graph)
labels(a dictionary of labels, e.g. {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'})
other options to determine node color node, etc

Returns: Q, maximum modularity and community_labels, the community assignments.
"""

def leiden(G,labels_dict,node_size=1000,font_size=17,resolution_parameter=.5,graph=False):
    #nx.write_graphml(G,'graph.graphml')
    #g = ig.read('graph.graphml',format="graphml")
    g = ig.Graph.Adjacency((nx.to_numpy_matrix(G) > 0).tolist())
    partition = la.find_partition(g, la.CPMVertexPartition,resolution_parameter = resolution_parameter)
    community_labels=[[labels_dict[member] for member in set(comm)] for comm in partition ]
    Q = modularity(G, partition)
    community_assignments=partition

    if graph:
        pos = nx.spring_layout(G)
        for comm in community_assignments:
            comm_node_list=comm
            nx.draw_networkx_nodes(G, pos, comm_node_list,node_size = node_size,node_color=color_names[list(community_assignments).index(comm)])
            nx.draw_networkx_edges(G, pos, alpha=0.5)
            nx.draw_networkx_labels(G,pos,labels_dict,font_size=font_size)
        ax=plt
        ax.axis('off')
        fig = ax.gcf()
        plt.title('Leiden Clustering',fontsize=18)
        plt.show()
    return community_assignments,Q,community_labels


"""
Laplacian: Form Laplacian matrix corresponding to a graph and plot eigenvalues

Argument: Undirected graph G

Returns Laplacian matrix L corresponding to G
"""

def Laplacian(G):
    L=nx.laplacian_matrix(G) #SciPy sparse matrix
    #print(L.todense())
    spectrum=nx.laplacian_spectrum(G)
    L=L.toarray()
    x=np.arange(len(spectrum))
    plt.scatter(x,spectrum)
    #spectrum=nx.laplacian_spectrum(G)
    #scipy.linalg.eig(L.toarray())
    #U, S, Vt = svds(L, k=2)
    return L

def Laplacian_normalized(G):
    import scipy
    L_norm=nx.normalized_laplacian_matrix(G).todense()
    spectrum,U=scipy.linalg.eig(L_norm)
    idx = spectrum.argsort()[::1]
    spectrum = spectrum[idx]
    U = U[:,idx]
    x=np.arange(len(spectrum))
    plt.scatter(x,spectrum)
    print(np.real(spectrum))
    return L_norm

"""
spectral communities: Determine communities by using eigenvectors
of k smallest eigenvalues of the Laplacian L corresponding to a graph.

Arguments: G (undirected, weighted graph)
k, number of communities
labels_dict, a dictionary of labels, e.g. {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'})
other options to determine node color node, etc

Returns: community labels
"""

def spectral_communities(G,k,labels_dict,node_size=1000,font_size=17,file_name='spectral_communities.png'):
  np.random.seed(0)
  sc = SpectralClustering(k, affinity='precomputed', n_init=100)
  A=nx.to_numpy_matrix(G, weight='weight')
  sc.fit(A)
  community_numbers = set(sc.labels_)
  community_assignments=[[y for y in np.where(sc.labels_==x)[0]] for x in community_numbers]
  community_labels=[[labels_dict[y] for y in np.where(sc.labels_==x)[0]] for x in community_numbers]
  pos = nx.spring_layout(G)
  for comm in community_assignments:
      comm_node_list=comm
      color_index=community_assignments.index(comm)
      nx.draw_networkx_nodes(G, pos, comm_node_list,node_size = node_size,node_color=color_names[color_index])
      nx.draw_networkx_edges(G, pos, alpha=0.5)
      nx.draw_networkx_labels(G,pos,labels_dict,font_size=font_size)
  ax=plt
  ax.axis('off')
  fig = ax.gcf()
  plt.title('Spectral Clustering',fontsize=17)
  plt.show()
  fig.savefig(file_name)
  return community_labels

"""
fuzzy_cmeans: Fuzzy c-means clustering based upon eigenvectors of # of k smallest eigenvalues of the
Laplacian L corresponding to a graph.

Arguments: G (undirected graph)
k, number of communities
max_iterations, is the maximum number of iterations
tolerance, specified tolerance.

Returns:
centers, luster centers.
membership_matrix, array of size N-by-k, where N denotes the number of nodes and row entries denote membership probabilities.


Unreturned values from fuzzy.cluster inclcude
u0, 2d array, (S, N) initial guess
d, array, finial Euclidean distance matrix
jm,1d array of length p, objective function history
p,number of iterations run
fpc,float final fuzzy partition coefficient
"""

def fuzzy_cmeans(G,k,max_iterations=10,tolerance=.01):
    L=nx.laplacian_matrix(G)
    L=L.toarray()
    D,Q=np.linalg.eig(L)
    data_points=Q[:,0:k].T
    centers, membership_matrix, u0, d, jm, p, fuzzy_partition_coeff= fuzzy.cluster.cmeans(data_points,k,2,error=tolerance,maxiter=max_iterations)
    membership_matrix=membership_matrix.T
    return centers, membership_matrix


small_world_indices: Returns small world indices as described in "How small is it? Comparing indices of
small worldliness," Network Science, (5), 30-44, 2017.

Arguments: G, niter (used for generating a random graph), nrand (number of random graphs)

Returns: Small world index Q, double-graphed normalized index, w (omega),
and small world index, SWI



def small_world_indices(G,niter=1,nrand=20):
    w=omega(G, niter=niter, nrand=nrand, seed=None)
    randMetrics = {"C_l": [], "L_l": [],"C_r": [], "L_r": []}
    for i in range(nrand):
        Gr = random_reference(G, niter=niter, seed=None)
        Gl = lattice_reference(G, niter=niter, seed=None)
        randMetrics["C_l"].append(nx.transitivity(Gl))
        randMetrics["L_l"].append(nx.average_shortest_path_length(Gl))
        randMetrics["C_r"].append(nx.transitivity(Gr))
        randMetrics["L_r"].append(nx.average_shortest_path_length(Gr))
    C = nx.transitivity(G)
    L = nx.average_shortest_path_length(G)
    C_l = np.mean(randMetrics["C_l"])
    L_l = np.mean(randMetrics["L_l"])
    C_r = np.mean(randMetrics["C_r"])
    L_r = np.mean(randMetrics["L_r"])
    Q=(C/C_r)/(L/L_r)
    SWI=(L-L_l)/(L_r-L_l)*(C-C_r)/(C_l-C_r)
    indices=[Q,w,SWI]
    return indices



'''




def Graph_communities_params(S,channels):

    import networkx as nx
    import networkx.algorithms.community as nc
    from networkx.algorithms.community.louvain import louvain_communities


    XKCD_COLORS = {
        'cloudy blue': '#acc2d9',
        'dark pastel green': '#56ae57',
        'dust': '#b2996e',
        'electric lime': '#a8ff04',
        'fresh green': '#69d84f',
        'light eggplant': '#894585',
        'nasty green': '#70b23f',
        'really light blue': '#d4ffff',
        'tea': '#65ab7c',
        'warm purple': '#952e8f',
        'yellowish tan': '#fcfc81',
        'cement': '#a5a391',
        'dark grass green': '#388004',
        'dusty teal': '#4c9085',
        'grey teal': '#5e9b8a',
        'macaroni and cheese': '#efb435',
        'pinkish tan': '#d99b82',
        'spruce': '#0a5f38',
        'strong blue': '#0c06f7',
        'toxic green': '#61de2a',
        'windows blue': '#3778bf',
        'blue blue': '#2242c7',
        'blue with a hint of purple': '#533cc6',
        'booger': '#9bb53c',
        'bright sea green': '#05ffa6',
        'dark green blue': '#1f6357',
        'deep turquoise': '#017374',
        'green teal': '#0cb577',
        'strong pink': '#ff0789',
        'bland': '#afa88b',
        'deep aqua': '#08787f',
        'lavender pink': '#dd85d7',
        'light moss green': '#a6c875',
        'light seafoam green': '#a7ffb5',
        'olive yellow': '#c2b709',
        'pig pink': '#e78ea5',
        'deep lilac': '#966ebd',
        'desert': '#ccad60',
        'dusty lavender': '#ac86a8',
        'purpley grey': '#947e94',
        'purply': '#983fb2',
        'candy pink': '#ff63e9',
        'light pastel green': '#b2fba5',
        'boring green': '#63b365',
        'kiwi green': '#8ee53f',
        'light grey green': '#b7e1a1',
        'orange pink': '#ff6f52',
        'tea green': '#bdf8a3',
        'very light brown': '#d3b683',
        'egg shell': '#fffcc4',
        'eggplant purple': '#430541',
        'powder pink': '#ffb2d0',
        'reddish grey': '#997570',
        'baby shit brown': '#ad900d',
        'liliac': '#c48efd',
        'stormy blue': '#507b9c',
        'ugly brown': '#7d7103',
        'custard': '#fffd78',
        'darkish pink': '#da467d',
        'deep brown': '#410200',
        'greenish beige': '#c9d179',
        'manilla': '#fffa86',
        'off blue': '#5684ae',
        'battleship grey': '#6b7c85',
        'browny green': '#6f6c0a',
        'bruise': '#7e4071',
        'kelley green': '#009337',
        'sickly yellow': '#d0e429',
        'sunny yellow': '#fff917',
        'azul': '#1d5dec',
        'darkgreen': '#054907',
        'green/yellow': '#b5ce08',
        'lichen': '#8fb67b',
        'light light green': '#c8ffb0',
        'pale gold': '#fdde6c',
        'sun yellow': '#ffdf22',
        'tan green': '#a9be70',
        'burple': '#6832e3',
        'butterscotch': '#fdb147',
        'toupe': '#c7ac7d',
        'dark cream': '#fff39a',
        'indian red': '#850e04',
        'light lavendar': '#efc0fe',
        'poison green': '#40fd14',
        'baby puke green': '#b6c406',
        'bright yellow green': '#9dff00',
        'charcoal grey': '#3c4142',
        'squash': '#f2ab15',
        'cinnamon': '#ac4f06',
        'light pea green': '#c4fe82',
        'radioactive green': '#2cfa1f',
        'raw sienna': '#9a6200',
        'baby purple': '#ca9bf7',
        'cocoa': '#875f42',
        'light royal blue': '#3a2efe',
        'orangeish': '#fd8d49',
        'rust brown': '#8b3103',
        'sand brown': '#cba560',
        'swamp': '#698339',
        'tealish green': '#0cdc73',
        'burnt siena': '#b75203',
        'camo': '#7f8f4e',
        'dusk blue': '#26538d',
        'fern': '#63a950',
        'old rose': '#c87f89',
        'pale light green': '#b1fc99',
        'peachy pink': '#ff9a8a',
        'rosy pink': '#f6688e',
        'light bluish green': '#76fda8',
        'light bright green': '#53fe5c',
        'light neon green': '#4efd54',
        'light seafoam': '#a0febf',
        'tiffany blue': '#7bf2da',
        'washed out green': '#bcf5a6',
        'browny orange': '#ca6b02',
        'nice blue': '#107ab0',
        'sapphire': '#2138ab',
        'greyish teal': '#719f91',
        'orangey yellow': '#fdb915',
        'parchment': '#fefcaf',
        'straw': '#fcf679',
        'very dark brown': '#1d0200',
        'terracota': '#cb6843',
        'ugly blue': '#31668a',
        'clear blue': '#247afd',
        'creme': '#ffffb6',
        'foam green': '#90fda9',
        'grey/green': '#86a17d',
        'light gold': '#fddc5c',
        'seafoam blue': '#78d1b6',
        'topaz': '#13bbaf',
        'violet pink': '#fb5ffc',
        'wintergreen': '#20f986',
        'yellow tan': '#ffe36e',
        'dark fuchsia': '#9d0759',
        'indigo blue': '#3a18b1',
        'light yellowish green': '#c2ff89',
        'pale magenta': '#d767ad',
        'rich purple': '#720058',
        'sunflower yellow': '#ffda03',
        'green/blue': '#01c08d',
        'leather': '#ac7434',
        'racing green': '#014600',
        'vivid purple': '#9900fa',
        'dark royal blue': '#02066f',
        'hazel': '#8e7618',
        'muted pink': '#d1768f',
        'booger green': '#96b403',
        'canary': '#fdff63',
        'cool grey': '#95a3a6',
        'dark taupe': '#7f684e',
        'darkish purple': '#751973',
        'true green': '#089404',
        'coral pink': '#ff6163',
        'dark sage': '#598556',
        'dark slate blue': '#214761',
        'flat blue': '#3c73a8',
        'mushroom': '#ba9e88',
        'rich blue': '#021bf9',
        'dirty purple': '#734a65',
        'greenblue': '#23c48b',
        'icky green': '#8fae22',
        'light khaki': '#e6f2a2',
        'warm blue': '#4b57db',
        'dark hot pink': '#d90166',
        'deep sea blue': '#015482',
        'carmine': '#9d0216',
        'dark yellow green': '#728f02',
        'pale peach': '#ffe5ad',
        'plum purple': '#4e0550',
        'golden rod': '#f9bc08',
        'neon red': '#ff073a',
        'old pink': '#c77986',
        'very pale blue': '#d6fffe',
        'blood orange': '#fe4b03',
        'grapefruit': '#fd5956',
        'sand yellow': '#fce166'}
    color_names=list(XKCD_COLORS.values())


    Nc=len(channels)
    labels_dict={i:channels[i] for i in range(Nc)}
    nodes=range(Nc)

    G=nx.from_numpy_matrix(S)
    clusters=louvain_communities(G)
    community_labels=[{channels[member] for member in comm} for comm in clusters ]
    node_to_community = dict()

    for i in range(Nc):
        for j in range(len(clusters)):
            if nodes[i] in clusters[j]:
                node_to_community.update({i:j})

    community_to_color = {i: color_names[i] for i in range(len(clusters))}
    node_color = {node: community_to_color[community_id] for node, community_id in node_to_community.items()}

    return G,community_labels,node_to_community,labels_dict,node_color
