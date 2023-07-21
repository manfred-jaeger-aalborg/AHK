# Contains a collection of synthetic data generators from simple AHK models
# and others

import numpy as np
import networkx as nx
from ahk import AHK_graphon
from utils import Signature

def data_colors(n,minnodes,maxnodes,coldis,**kwargs):
    # Generates graphs with a single unary "color" attribute and
    # between minnodes and maxnodes many nodes
    # n: number of graphs
    # coldis: probability vector of color values


    # Defining the model:
    sig = Signature([len(coldis)],0,directed=False)
    
    binbounds=np.zeros(len(coldis)+1)
    for i in range(len(coldis)):
        binbounds[i+1]=binbounds[i]+coldis[i]
    binbounds[-1]=1.0
    
    ahk=AHK_graphon(sig,binbounds)
    f1=[np.identity(len(coldis))]
    ahk.set_f1(f1)

    # Sampling from the model:
    data=[]
    for i in range(n):
        nn=np.random.randint(minnodes,high=maxnodes)
        data.append(ahk.sample_world(nn,**kwargs))


    return data

def ahk_sbm(coldis,edgeprobs,directed):
    sig = Signature([len(coldis)],1,directed=directed)
    binbounds=np.zeros(len(coldis)+1)
    for i in range(len(coldis)):
        binbounds[i+1]=binbounds[i]+coldis[i]
    binbounds[-1]=1.0

    ahk=AHK_graphon(sig,binbounds)
    f1=[np.identity(len(coldis))]
    ahk.set_f1(f1)

    if directed:
        ff=np.zeros((ahk.granularity,ahk.granularity,1,2))
        for i in range(ahk.granularity):
            for j in range(i,ahk.granularity):
                ff[i,j,0,0]=edgeprobs[i,j]
                ff[i,j,0,1]=edgeprobs[j,i]
        
    if not directed:
        ff=np.zeros((ahk.granularity,ahk.granularity,1))
        for i in range(ahk.granularity):
            for j in range(i,ahk.granularity):
                ff[i,j,0]=edgeprobs[i,j]
    ahk.set_f2(ff)
    return ahk
    

def sample_data(model,n,minnodes,maxnodes,**kwargs):
    data=[]
    for i in range(n):
        nn=np.random.randint(minnodes,high=maxnodes)
        data.append(model.sample_world(nn,**kwargs))
    return data


# Generating 'spatial' random graphs:

def random_spatial_graph(n,q,directed):
    if directed:
        G=nx.DiGraph()
    else:
        G=nx.Graph()
    positions={}
    for i in range(n):
        pos=np.random.random(2)
        G.add_node(i)
        G.nodes[i]['pos']=pos
        positions[i]=pos
    for i in range(n):
        if directed:
            jrange=np.hstack((np.array(range(i),dtype=int),np.array(range(i+1,n),dtype=int)))
        else:
            jrange=range(i+1,n)
        for j in jrange:
            if np.random.random()< \
               np.exp(-q*np.linalg.norm(positions[i]-positions[j])):
                G.add_edge(i,j)
    return G,positions

# Simulating information cascade:

def initCascade(G,numseeds):
    for i in range(len(G.nodes())):
        G.nodes[i]['seed']=0
        G.nodes[i]['infected']=0
        G.nodes[i]['activated']=0

    seeds=np.random.choice(range(len(G.nodes())),size=numseeds,replace=False)

    for i in seeds:
        G.nodes[i]['seed']=1
        G.nodes[i]['activated']=1
        G.nodes[i]['infected']=1
    return

def propagate(G,p): 
    # p is the propagation probability
    newactive=False
    for n,neigh in G.adjacency():
        if G.nodes[n]['activated']==1:
            G.nodes[n]['activated']==0
            for nn in neigh:
                if G.nodes[nn]['infected']==0:
                    if np.random.rand()<p:
                        G.nodes[nn]['activated']=1
                        G.nodes[nn]['infected']=1
                        newactive=True
    return newactive
     
def make_Cascade(G,numseeds,p):
    initCascade(G,numseeds)
    done = False
    while not done:
        done = propagate(G,p)
    return

def sample_egos(G,n_sample,radius,**kwargs):
    # G a nx graph
    egos=[]
    nodes=list(G.nodes())
    for j in range(n_sample):
        i=np.random.randint(len(nodes))
        # kwargs especially: undirected=True means that for a directed graph both
        # in- and out-neighbors are included (otherwise only out-neighbors)
        egos.append((nodes[i],nx.ego_graph(G,nodes[i],radius=radius,**kwargs)))
    return egos

def sample_egos_from_batch(Gs,n_sample,radius):
    # Gs: a list of nx graphs
    egos=[]
    for G in Gs:
        egos+=sample_egos(G,n_sample,radius)
    return egos
