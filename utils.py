import numpy as np
import networkx as nx
import math

class Signature():
    def __init__(self,unaries,nb,directed=True):
        self.unaries = unaries # array of number of possible values of attributes: e.g. unaries=[3,5]: two
                               # attributes, the first of which with 3, the second with 5 possible values
        self.nb = nb # number of binary relations (no self-loops allowed)
        self.directed = directed # Boolean; whether all binary relations are directed; mix of directed and 
                                 # undirected edges is not supported
        
    def numatoms_binary(self,n): # number of binary atoms for domain of size n
                                 # All binary relations are required to be withoug self-loops (existence of
                                 # such a loop could be represented by a special attribute)
        if not self.directed:
            return self.nb*n*(n-1)
        else:
            return self.nb*n*(n-1)/2
        
    def numatoms_unary(self,n): # number of unary atoms for domain of size n
        nu=0
        for i in range(len(self.unaries)):
            nu+=n*self.unaries[i]
        return nu
    
    def numatoms(self,n): # number of atoms for domain of size n
        return self.numatoms_unary(n) + self.numatoms_binary(n)
    
    def numworlds(self,n): # number of worlds of size n
        return 2**self.numatoms(n)
        
class World():
    def __init__(self,sig,n):
        self.directed = sig.directed
        self.sig=sig
        self.n = n
        self.unaries=np.zeros((n,len(sig.unaries)),dtype=int) 
        self.binaries=np.zeros((n,n,sig.nb),dtype=int)
        
    def setatt(self,k,A):
        # A: value vector of length n with values for k'th attribute
        self.unaries[:,k]=A            
        
    def setatts(self,i,atts):
        # sets all attribute values for node i
        self.unaries[i,:]=atts
        
    def setrel(self,k,A):
        if not self.directed:
               # set diagonal and below to zero
               A=np.triu(A,1)
        else:
            # set diagonal to zero:
            np.fill_diagonal(A, 0)
        self.binaries[:,:,k]=A
 
    def setatom(self,k,arg,val):
        if len(arg)==1: # unary
            self.unaries[arg,k]=val
        if len(arg)==2:
            self.binaries[arg[0],arg[1],k]=val

    def typeCounts_unary(self,tcounts):
        # returns a dictionary type:count
        # if tcounts != None, then counts of this world are added to
        # counts in tcounts
        if tcounts == None:
            tcounts={}
        
        for i in range(self.n):
            key=str(self.unaries[i,:])
            if key in tcounts:
                tcounts[key]+=1
            else:
                tcounts[key]=1
        return tcounts
        
    def typeCounts_binary(self,tcounts):
        # returns a dictionary type:count
        # if tcounts != None, then counts of this world are added to
        # counts in tcounts
        if tcounts == None:
            tcounts={}
        
        for i in range(self.n-1):
            for j in range(i+1,self.n):
                key=str(self.unaries[i,:])+"_"+str(self.unaries[j,:])+"_"+str(self.binaries[i,j,:])+"_"+str(self.binaries[j,i,:])
                if key in tcounts:
                    tcounts[key]+=1
                else: 
                    tcounts[key]=1
        return tcounts

        
    def to_nx(self):
        if self.directed:
            G = nx.MultiDiGraph()
        else:
            G = nx.MultiGraph()
        for i in range(self.n):
            G.add_node(i,features=self.unaries[i,:])
        for r in range(self.sig.nb):
            edidxs=np.where(self.binaries[:,:,r]==1)
            for e in range(len(edidxs[0])):
                G.add_edge(edidxs[0][e],edidxs[1][e],rel="b"+str(r))    
        return(G)    
   
    def subsample(self,k):    
        """
        Sample a sub-world of size k. 
        """
        sw=World(self.sig,k)
        
        nodes=np.sort(np.random.choice(range(self.n),size=k,replace=False))
        
        sw.unaries=self.unaries[nodes,:]
        
        rowsel=self.binaries[nodes,:,:]
        sw.binaries=rowsel[:,nodes,:]
        
        return sw

def enumerate_unary_worlds(sig):
    # return a list of all one-node worlds in signature sig (only unary attributes matter)
    unarytypevecs=all_un_vecs(sig.unaries)
    worlds=[]
    for r in range(unarytypevecs.shape[0]):
        nextw=World(sig,1)
        nextw.setatts(0,unarytypevecs[r,:])
        worlds.append(nextw)
    return worlds
    
def normalize_typeCounts(tcounts):
    sum =0
    for v in tcounts.values():
        sum+=v
    for k in tcounts.keys():
        tcounts[k]/=sum
    return tcounts


def nx_to_world(nxg,  featmaps=None):
    """
    Translates an undirected networkx graph nxg into a world.

    featmaps is a dictionary that contains for (selected) node attributes
    of nxg a mapping from attribute values to integers 0,...,k
    """

    unaries=()
    
    if not featmaps==None:
        numatts=len(featmaps.keys())
        unaries=np.zeros(numatts,dtype=int)
        for i,k in enumerate(featmaps.keys()):
            print(i,k)
            unaries[i]=len(featmaps[k].keys())
    
    sig=Signature(unaries,1,nx.is_directed(nxg))
    
    w=World(sig,nx.number_of_nodes(nxg))
   
    if not featmaps==None:
        attvecs=[]
        for node in nxg.nodes():
            nodeatt=np.zeros(numatts,dtype=int)
            for i,k in enumerate(featmaps.keys()):
                nodeatt[i]=nxg.nodes[node][k]
            attvecs.append(nodeatt)
        w.unaries=np.vstack(attvecs)
        
    w.binaries[:,:,0]=nx.adjacency_matrix(nxg).todense()
    
    return w


def batch_nx_to_world(nx_data, featmaps=None):
    worlds=[]
    for wnx in nx_data:
        worlds.append(nx_to_world(wnx,featmaps=featmaps))
    return worlds

def uni_bins(g):
    result=np.zeros(g+1)
    for i in range(1,g):
        result[i]=i*(1/g)
    result[g]=1.0    
    return result


def cat_to_int(c):
    """"
    transform array of string values into integer array
    """
    intarr=np.zeros(len(c),dtype=int)
    nextint=0
    dict={}
    for s in range(len(c)):
        if c[s] not in dict.keys():
            dict[c[s]]=nextint
            nextint+=1
        intarr[s]=dict[c[s]]

    return dict,intarr

def cat_to_int_dict(c,dict):
    """
    as cat_to_int but with a fixed dictionary
    """
    intarr=np.zeros(len(c),dtype=int)
 
    for s in range(len(c)):
        intarr[s]=dict[c[s]]

    return intarr



def get_att_array(G,att_name):
    ret_array=np.zeros(nx.number_of_nodes(G))
    for i,n in enumerate(G.nodes()):
        ret_array[i]=G.nodes[n][att_name]
    return(ret_array)

def all_un_vecs(unaries):
    # returns a matrix whose rows contain all possible combinations of values for the unary relations in sig
    if len(unaries)==1:
        return np.array(range(unaries[0]),dtype=int).reshape(-1,1)
    tailmatrix=all_un_vecs(unaries[1:])
    result=np.hstack((np.zeros(tailmatrix.shape[0],dtype=int).reshape(-1,1),tailmatrix))
    for k in range(1,unaries[0]):
        nextblock=np.hstack((np.full(tailmatrix.shape[0],k,dtype=int).reshape(-1,1),tailmatrix))
        result=np.vstack((result,nextblock))
    return result

def data_typeCounts_binary(data):
    tcounts={}
    for d in data:
        tcounts=d.typeCounts_binary(tcounts)     
    tcounts=normalize_typeCounts(tcounts)
    return tcounts

def data_typeCounts_unary(data):
    tcounts={}
    for d in data:
        tcounts=d.typeCounts_unary(tcounts)     
    tcounts=normalize_typeCounts(tcounts)
    return tcounts

def splitprobs(p):
    """
    Splits a probability distribution p=p0,...,p(n-1) randomly into two distributions
    q0=q00,...,q0(n-1), q1=q10,...,q1(n-1), such that 0.5*q0+0.5*q1=p
    """
    # randomly generate a direction:
    d=np.random.random(len(p))-0.5
    print("d 1: ", d)
    d=d*(np.minimum(p,1-p)) #smaller changes for coordinates near 1 or 0
    print("d 2: ", d)
    d=d/np.sum(d)
    
    # determine distance from p to the boundary of the probability parameter space
    # in directions d and -d: (-600-)

    
    l=np.min((1-p)/np.abs(d))
    l=np.min((l,np.min(p/np.abs(d))))

    print("d: ", d, "l: ",l)
    
    #random steplength:
    sl=np.random.random()*l

    return p+sl*d,p-sl*d


def split3(p):
    """
    Splits probability value p into q1,q2,q3, such that p=0.5*q1+0.25*q2+0.25*q2
    """
    minq1=np.max((0,p-0.25))
    maxq1=np.min((1,2*p))
    q1=minq1+(maxq1-minq1)*np.random.random()
    
    sumq2q3=4*(p-0.5*q1)
    minq2=np.max((0,sumq2q3-1))
    maxq2=np.min((1,sumq2q3))

    q2=minq2+(maxq2-minq2)*np.random.random()
    
    q3=4*(p-0.5*q1-0.25*q2)
    return np.array((q1,q2,q3))
    
