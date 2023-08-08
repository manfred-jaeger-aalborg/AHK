import numpy as np
import networkx as nx
import math

from utils import World,Signature,enumerate_unary_worlds


import itertools
from itertools import permutations,combinations_with_replacement,product,combinations
from scipy.special import binom

from tqdm.auto import tqdm


class AHK_graphon():
    def __init__(self,signature,binbounds,minus=True):
        # binbounds is an array b of the form 0=b[0]<b[1]<...<b[g-1]<b[g]=1
        # g then is the granularity, i.e., number of bins
        self.minus = minus # whether this is a AHK^- model
        self.binbounds=binbounds
        self.granularity=  len(binbounds)-1 # number of partition elements (common for all u variables)
        self.signature = signature
        self.directed=signature.directed
        self.attrange=range(len(self.signature.unaries))
        
        """
        for self.minus=False:
        f1[k][i,j] contains the probability for a_k(a)=j if 
        -- a_k is the k'th unary relation
        -- a is node whose u value u_a falls into the i'th bin
        -- j is the j'th value of a_k
        
        f2[i,j,k,0] (resp.f2[i,j,k,1]) contains the probability for e_k(a,b)=true (resp. e_k(b,a)=true), if
        -- e_k is the k'th binary relation
        -- a,b are nodes whose u values u_a,u_b are in the i'th and j'th bin, respectively
        -- u_a<u_b. This implies that i<=j, and the lower triangle of this matrix always is empty
        -- if undirected, then no last argument 
        
        for self.minus=True the additional first coordinate represents the binned value of
        the global U_\emptyset variable (not implemented yet)
        """
        if self.minus:
            self.f1 = list(np.zeros((self.granularity,self.signature.unaries[i])) for i in self.attrange)
            if self.directed:
                self.f2 = np.zeros((self.granularity,self.granularity,self.signature.nb,2))
            else:
                self.f2 = np.zeros((self.granularity,self.granularity,self.signature.nb))
        else:
            self.f1 = list(np.zeros((self.granularity,self.granularity,self.signature.unaries[i])\
                                    for i in self.attrange))
            if self.directed:
                self.f2 = np.zeros((self.granularity,self.granularity,self.granularity,self.signature.nb,2))
            else:
                self.f2 = np.zeros((self.granularity,self.granularity,self.granularity,self.signature.nb))

    def copy(self):
        mycopy=AHK_graphon(self.signature,self.binbounds.copy(),self.minus)
        mycopy.f1=list(self.f1[i].copy() for i in self.attrange)
        mycopy.f2=self.f2.copy()
        return mycopy
    
    def get_binlengths(self):
        return np.array(list(self.binbounds[i]-self.binbounds[i-1] for i in range(1,len(self.binbounds))))
        
        
    def rand_init(self):
        for i in range(len(self.f1)):
            self.f1[i]=np.random.random(self.f1[i].shape)
            for g in range(self.granularity):
                self.f1[i][g,:]=self.f1[i][g,:]/np.sum(self.f1[i][g,:])
        self.f2=np.random.random(self.f2.shape)
        for k in range(self.signature.nb):
            if self.directed:
                self.f2[:,:,k,0]=np.triu(self.f2[:,:,k,0])
                self.f2[:,:,k,1]=np.triu(self.f2[:,:,k,1])
            else:
                self.f2[:,:,k]=np.triu(self.f2[:,:,k])
        

    def set_f1(self,f):
        self.f1=f

    def set_f2(self,f):
        self.f2=f

    def get_one_type_dist(self):
        # returns the distribution over one-types defined by this model
        # returns a dictionary with type keys consistent with utils.typeCounts_unary
        one_worlds=enumerate_unary_worlds(self.signature)
        result={}
        for w in one_worlds:
            key = ""
            for i in self.attrange:
                key=key+"["+str(w.unaries[i])+"]"
                result[key]=self.compute_prob(w)
        return result
        
    def btuples(self,k): # if self.minus: enumerates all b_1<=b_2<= ... <= b_k
                         # if not self.minus: b_0,b_1<=b_2<= ... <= b_k, where all b's 
                         # range from 0 to granularity-1
        if self.minus:
            return combinations_with_replacement(range(self.granularity),k)
        else:
            return product(range(self.granularity),combinations_with_replacement(range(self.granularity),k))
       
    def get_binindx(self,u):
        return np.min(np.where(self.binbounds>u))-1
        
    def volume(self,b,withgradient=False): 
        """
        For a bin sequence b=(b1,b2,...,bk) with b1<=b2<=...<=bk computes the volume of the
        cartesian product b1 x b2 x ... x bk intersected with the 'triangle' 
        (u1,u2,...,uk) \in [0,1]^k: u1<=u2<=...<=uk
        
        cf. -537-,-592-
        """
        
        runcounts=np.zeros(self.granularity)
        
        v=self.binbounds[b[0]+1]-self.binbounds[b[0]]
        run=1
        grad=None
        
        for i in range(1,len(b)):
            v*=(self.binbounds[b[i]+1]-self.binbounds[b[i]])
            if b[i]==b[i-1]:
                run+=1
            else:
                runcounts[b[i-1]]=run
                v*=1/math.factorial(run)
                run=1
        v*=1/math.factorial(run)
        runcounts[b[-1]]=run
        
        if withgradient:
            grad=np.zeros(self.granularity+1) #first and last component are gradients for the fixed bounds
                                              # 0,1, and therefore always 0
            for i in range(1,self.granularity):
                grad[i]+=runcounts[i-1]/(self.binbounds[i]-self.binbounds[i-1])
                grad[i]-=runcounts[i]/(self.binbounds[i+1]-self.binbounds[i])
            grad*=v
        return v,grad


    def get_bprobs(self,n):
        """
        Returns vector of volumes of all b-vectors of length n.
        This is equal to the probability P(b,pi), for any pi.
        """
        bvolumes=np.zeros(math.comb(self.granularity+n-1,n))
        for bidx,b in enumerate(self.btuples(n)):
            bvolumes[bidx]=self.volume(b)[0] 
        #bvolumes=bvolumes/np.sum(bvolumes)
       
        return bvolumes
    
    """
    Calculate the probability of the unary type of node n in world w if it belongs to bin b
    """
    def getprob_unary_node(self,b,n,w):
        result=1
        for att in self.attrange:
            if not w.unaries[n,att]==-1:
                result*=self.f1[att][b,w.unaries[n,att]]
        return result
     
    """
    Calculate the probability of the binary type of node pair n1,n2 if they belong to bins b1 <= b2
    (and it is assumed that pi^{-1}(n1)<pi^{-1}(n2))
    """    
    def getprob_binary_nodes(self,b1,b2,n1,n2,w):
        assert b1<=b2
        result=1
        if self.directed:
            for rel in range(self.signature.nb):
                if w.binaries[n1,n2,rel]==1:
                    result*=self.f2[b1,b2,rel,0]
                if w.binaries[n1,n2,rel]==0:
                    result*=1-self.f2[b1,b2,rel,0]
                if w.binaries[n2,n1,rel]==1:
                    result*=self.f2[b1,b2,rel,1]
                if w.binaries[n2,n1,rel]==0:
                    result*=1-self.f2[b1,b2,rel,1]
        else:
            assert n1<n2
            for rel in range(self.signature.nb):
                if w.binaries[n1,n2,rel]==1:
                    result*=self.f2[b1,b2,rel]
                if w.binaries[n1,n2,rel]==0:
                    result*=1-self.f2[b1,b2,rel]
        return result  
    
    def precompute_single_factors(self,w):
        result=np.zeros((self.granularity,w.n))
        for i in range(w.n):
            for b in range(self.granularity):
                    result[b,i]=self.getprob_unary_node(b,i,w)
        return result
    
    """
    Collects all getprob_binary_nodes(b1,b2,n1,n2,w) in one matrix
    """
    def precompute_pair_factors(self,w):
        result=np.zeros((self.granularity,self.granularity,w.n,w.n))
        if self.directed:
            for (n1,n2) in permutations(range(w.n),r=2): #iterating over all pairs of nodes n1 != n2
                for b1 in range(self.granularity): 
                    for b2 in range(b1,self.granularity):
                        result[b1,b2,n1,n2]=self.getprob_binary_nodes(b1,b2,n1,n2,w)
        else:
            for (n1,n2) in combinations(range(w.n),2): #iterating over all pairs of nodes n1 < n2
                for b1 in range(self.granularity): 
                    for b2 in range(b1,self.granularity):
                        result[b1,b2,n1,n2]=self.getprob_binary_nodes(b1,b2,n1,n2,w)
                        result[b1,b2,n2,n1]= result[b1,b2,n1,n2]
        return result
    
    
    def calculate_pi_b_precomp(self,pi,b,UP,BP):
        # UP,BP arguments are results of precompute_single_factors and precompute_pair_factors
        # Returns the probability of the world for which UP and BP have been computed given 
        # permutation pi and bin assignment b
        prod=1
        # unary relations:
        for p in range(UP.shape[1]):
            prod*=UP[b[p],pi[p]]

        # binary relations:
        for (p,q) in combinations(range(UP.shape[1]),2): # iterating over pairs of nodes
                prod*=BP[b[p],b[q],pi[p],pi[q]]
        return prod     
    

 
    
    def compute_prob(self,w,**kwargs): 
        """
        Exact inference. Only for very small w!
        """
        
        if 'query' in kwargs.keys():
            qs=kwargs['query']
            tq=np.zeros(len(qs))
            
        else:
            qs=None
            
        p=0
        
        UP=self.precompute_single_factors(w)
        BP=self.precompute_pair_factors(w)
        
              
        P_of_pi_b=self.get_bprobs(w.n)
        
        for pi in itertools.permutations(range(w.n)):
            #print("\n pi: {}".format(pi))
            pi=np.array(pi)
            ppi=0
            for bidx,b in enumerate(self.btuples(w.n)):
                b=np.array(b)
                pi_b_prob=P_of_pi_b[bidx]*self.calculate_pi_b_precomp(pi,b,UP,BP)
                
                ppi+=pi_b_prob
          
                if not qs==None:
                    for q in range(len(qs)):
                        qprob=1 #conditional probability of query qs[q]
                        for at in qs[q]:
                            if len(at)==3:
                                bofarg=b[np.where(pi==at[1])]
                                qfac=self.f1[at[0]][bofarg,at[2]]

                            if len(at)==4:
                                piidx1=np.where(pi==at[1])
                                piidx2=np.where(pi==at[2])
                                bofarg1=b[piidx1]
                                bofarg2=b[piidx2]
                                if self.directed:
                                    if piidx1<piidx2:
                                        qfac=self.f2[bofarg1,bofarg2,at[0],0]
                                    else:
                                        qfac=self.f2[bofarg2,bofarg1,at[0],1]
                                else:
                                    if piidx1<piidx2:
                                        qfac=self.f2[bofarg1,bofarg2,at[0]]
                                    else:
                                        qfac=self.f2[bofarg2,bofarg1,at[0]]
                                if not at[-1]:
                                    qfac=1-qfac
                            qprob*=qfac
                        tq[q]+=qprob*pi_b_prob
                    
            p+=ppi

        if not qs==None:
            return tq/p,p
        else:
            return p
      
    # def compute_pi_b_matrix(self,w): 


    #     UP=self.precompute_single_factors(w)
    #     BP=self.precompute_pair_factors(w)
        
    #     bvolumes=self.get_bprobs(w.n)

    #     result=np.zeros((math.factorial(w.n),int(binom(w.n+self.granularity-1,w.n))))
    #     for r,pi in enumerate(itertools.permutations(range(w.n))):
    #         for c,b in enumerate(self.btuples(w.n)):
    #             result[r,c]=bvolumes[c]*self.calculate_pi_b_precomp(pi,b,UP,BP)
    #     return result

    
    # def compute_pi_b_grad_matrix(self,w,params): 

    #     """
    #     Limited function for debugging
    #     """
    #     bvolumes=self.get_bprobs(w.n)

    #     result={}
    #     for p in params:
    #         result[p]=np.zeros((math.factorial(w.n),int(binom(w.n+self.granularity-1,w.n))))
      
    #     for r,pi in enumerate(itertools.permutations(range(w.n))):
    #         for c,b in enumerate(self.btuples(w.n)):
    #             g1,g2,gv=self.compute_grad(w,pi,b)
    #             for p in params:
    #                 result[p][r,c]=bvolumes[c]*g1[0][p[0],p[1]]
                    
    #     return result
    
    def pi_to_index(self,n):
        """
        Computes dictionary that maps permutations of (0,...,n-1) to the index of 
        its position in the enumeration by itertools.permutations(range(n))
        """
        result = {}
        
        for i,pi in enumerate(itertools.permutations(range(n))):
            result[tuple(pi)]=i
            
        return result
            
        
    def index_to_pi(self,n):
        result = {}
        
        for i,pi in enumerate(itertools.permutations(range(n))):
            result[i]=pi
            
        return result
    
    
    def b_to_index(self,n):
        result = {}
        
        for i,b in enumerate(self.btuples(n)):
            result[b]=i
            
        return result
            
    def index_to_b(self,n):
        result = {}
        
        for i,b in enumerate(self.btuples(n)):
            result[i]=bb
            
        return result
 
    def get_piweights(self,UP,BP):
        # construct a weight vector for use in random_pi
        
        # First compute matrix with elements BP[k,h,i,j]*UP[k,i]*UP[h,j]:
        piweights=np.moveaxis(BP,2,1)*UP
        piweights=np.moveaxis(piweights,(2,3),(0,1))*UP
        # This has now shape [g,w.n,g,w.n] with arguments [h,j,k,i]
        # marginalizaing out the bin arguments and swapping again the order of j,i:
        piweights=np.transpose(np.sum(piweights,axis=(0,2)))

        return piweights

    def get_bweights(self,BP):
         #bweights = np.sum(BP,axis=1)
        bweights = np.max(BP,axis=1)
        return bweights
    
    def importance_sample(self,w,return_pibs=False,**kwargs):
        """
        w: an incompletely observed world
        possible keyword args:
        
        query=q:

        q a list of queries, where each query is a list of query atoms given by:
        -- triples (r,i,j) with: r: index of an attribute,
                                 i: node index
                                 j: possible value of r'th attribute
        or
        --  quadruples (r,i,j,True/False) with r index of a binary relation
                                               i,j node indices.

        Computes the probability of the joint truth assignments to all query atoms.

        If no query is 
        specified, then compute marginal probability of w
        
        numsamples=n: the (fixed) number of samples that are generated
        
        A=n, epsilon=e termination criterion is defined by: stop when in the last n samples there 
        was no greater relative change than epsilon in the estimated probability
        
        Either numsamples or A and epsilon must be specified
        """
        
        if 'query' in kwargs.keys():
            qs=kwargs['query']
        else:
            qs=None
            
        if 'numsamples' in kwargs.keys():
            numsamples=kwargs['numsamples']
        else:
            numsamples=None
            if 'A' in kwargs.keys() and 'epsilon' in kwargs.keys():
                A=kwargs['A']
                epsilon=kwargs['epsilon']
            else:
                print("Lacking termination condition in importance_sample. Specify either 'numsamples' or 'A' and 'epsilon'")
            
        UP=self.precompute_single_factors(w)
        BP=self.precompute_pair_factors(w)
                
        piweights=self.get_piweights(UP,BP)
        bweights=self.get_bweights(BP)
            
        P_of_pi_b=self.get_bprobs(w.n)
        bidx=self.b_to_index(w.n)
            
        terminate=False
        sampled = 0
        pw=0 # sum of weights of sampled worlds

        if not qs==None:
            tq=np.zeros(len(qs)) # sum of probabilities for query=true
            fq=np.zeros(len(qs)) # sum of probabilities for query=false
        
        nochangecount=0
        lastvalue=1
        trace=[]

        num_pibs=None
        if return_pibs:
             numus=math.comb(self.granularity+w.n-1,w.n)
             b_to_index=self.b_to_index(w.n)
             numpis=math.factorial(w.n)
             pi_to_index=self.pi_to_index(w.n)
             num_pibs=np.zeros((numpis,numus))
        
        while not terminate:
            sampled+=1
            
            piprob,pi=self.random_pi(w.n,piweights)
            #print("pi: ", pi)
             
            bprob,b=self.random_b_for_pi(pi,BP,UP,bweights)
            #print("b: ", b)
            
            if return_pibs:
                num_pibs[pi_to_index[tuple(pi)],b_to_index[tuple(b)]]+=1
      
            p_pi_b=self.calculate_pi_b_precomp(pi,b,UP,BP)
            worldweight=p_pi_b*P_of_pi_b[bidx[tuple(b)]]/(piprob*bprob)
            pw+=worldweight
            
            if not qs==None:
                for q in range(len(qs)):
                    qprob=1
                    for at in qs[q]:
                        if len(at)==3:
                            bofarg=b[np.where(pi==at[1])]
                            qfac=self.f1[at[0]][bofarg,at[2]]

                        if len(at)==4:
                            piidx1=np.where(pi==at[1])
                            piidx2=np.where(pi==at[2])
                            bofarg1=b[piidx1]
                            bofarg2=b[piidx2]
                            if self.directed:
                                if piidx1<piidx2:
                                    qfac=self.f2[bofarg1,bofarg2,at[0],0]
                                else:
                                    qfac=self.f2[bofarg2,bofarg1,at[0],1]
                            else:
                                if piidx1<piidx2:
                                    qfac=self.f2[bofarg1,bofarg2,at[0]]
                                else:
                                    qfac=self.f2[bofarg2,bofarg1,at[0]]
                            if not at[-1]:
                                qfac=1-qfac
                        qprob*=qfac

                    tq[q]+=qprob*worldweight
                    fq[q]+=(1-qprob)*worldweight
                trace.append(tq/(tq+fq))
            else:
                trace.append(pw/sampled)
            # now determine termination:
            if not numsamples==None:
                terminate=(sampled==numsamples)
            else: # this termination criterion not yet adapted to multiple queries!
                if not q==None:
                    if tq>0:
                        thisval=tq/(tq+fq)
                    else:
                        thisval=0
                else:
                    thisval=pw/sampled
                if thisval>0 and lastvalue>0:
                    ratio=thisval/lastvalue
                    if ratio < 1+epsilon and ratio>1-epsilon:
                        nochangecount+=1
                    else:
                        nochangecount=0
                else:
                    if thisval==0 and lastvalue==0:
                        nochangecount+=1
                    else:
                        nochangecount=0
                lastvalue=thisval
                terminate=(nochangecount==A)
               
            
        if not qs==None:
            return tq/(tq+fq),trace,num_pibs
        else:
            return pw/sampled,trace,num_pibs
       
        
        
    def random_pi(self,n,weights):
        #return np.random.permutation(n)
        """
        weights is an nxn matrix. The entries weights[i,j] and weights[j,i] represent 
        relative weights for putting elements i,j in the order i<j, resp. j<i
        """
        
        #initpi=np.array(range(n)) 
        initpi=np.random.permutation(n) 
        rpi=[]
        piprob=1
        # first two elements:
        i=initpi[0]
        j=initpi[1]
        prob=weights[i,j]/(weights[i,j]+weights[j,i])
        if np.random.random()<prob:
            rpi.append(i)
            rpi.append(j)
            piprob*=prob
        else:
            rpi.append(j)
            rpi.append(i)
            piprob*=(1-prob)
        
        for l in range(2,n):
            i=initpi[l]
            beforeweights=np.ones(len(rpi)+1)
            afterweights=np.ones(len(rpi)+1)
            for k in range(len(rpi)):
                beforeweights[k+1]=beforeweights[k]*weights[rpi[k],i]
                afterweights[-k-2]=afterweights[-k-1]*weights[i,rpi[-k-1]]
            probs=beforeweights*afterweights
            psum=np.sum(probs)
            if psum>0:
                probs=probs/np.sum(probs)
            else:
                probs=np.ones(len(rpi)+1)/(len(rpi)+1) # in this case no useful sample will be obtained
            ii=np.arange(len(rpi)+1)
            idx=np.random.choice(ii,p=probs)
            
            rpi.insert(idx,i)
            piprob*=probs[idx]
    
        return piprob,np.array(rpi)
            
            
        
    def random_b_for_pi(self,pi,BP,UP,bweights):
        """
        bweights is a (granularity x n x n ) matrix containing for each pair (i,j) of nodes and each u-bin 
        a weight
        """
        rb=np.zeros(len(pi),dtype=np.int32)
        bp=1
        
        bvals=np.arange(self.granularity) # provides the sample space for possible u values

        #print("b for pi: ", pi)
        for p in range(len(rb)):
            bprobs=UP[:,pi[p]].copy()
            #print("bprobs #1:", bprobs)
            for q in range(p):
                bprobs*=BP[rb[q],:,pi[q],pi[p]]
                #print("bprobs #2:", bprobs, "mult. with " , BP[rb[q],:,pi[q],pi[p]])
            for q in range(p+1,len(rb)):
                bprobs*=bweights[:,pi[p],pi[q]]
                #print("bprobs #3:", bprobs)
            probsum=np.sum(bprobs)
            if probsum >0:
                bprobs=bprobs/np.sum(bprobs)
            else:
                bprobs=np.hstack((np.zeros(rb[p-1]),\
                                  np.ones(self.granularity-rb[p-1])/(self.granularity-rb[p-1])))
            #print("bprobs #4:", bprobs)
            sampled=np.random.choice(bvals,p=bprobs)
            rb[p]=sampled
            bp*=bprobs[sampled]
            
        return bp,rb
        
    def sample_pi_and_b(self,w,numsamps):
        """
        Sample (pi,u) pairs, and return in a matrix with counts for the different (pi,u) 
        combinations. For debugging and illustrative purposes (this is basically running importance 
        sampling, without the probability computations)
        """
        UP=self.precompute_single_factors(w)
        BP=self.precompute_pair_factors(w)
        
 
        piweights=np.moveaxis(BP,2,1)*UP
        piweights=np.moveaxis(piweights,(2,3),(0,1))*UP
        # This has now shape [g,w.n,g,w.n] with arguments [h,j,k,i]
        # marginalizaing out the bin arguments and swapping again the order of j,i:
        piweights=np.transpose(np.sum(piweights,axis=(0,2)))
        
        bweights = np.sum(BP,axis=1)
        
  
        numus=math.comb(self.granularity+w.n-1,w.n)
        b_to_index=self.b_to_index(w.n)
       

        numpis=math.factorial(w.n)
        pi_to_index=self.pi_to_index(w.n)

            
        importance_sample=np.zeros((numpis,numus))

        for i in range(numsamps):          
            piprob,pi=self.random_pi(w.n,piweights)
            bprob,b=self.random_b_for_pi(pi,BP,UP,bweights)
            importance_sample[pi_to_index[tuple(pi)],b_to_index[tuple(b)]]+=1
        return importance_sample
        
    def compute_loglik(self,data):
        loglik=0
        for w in data:
            loglik+=np.log(self.compute_prob(w)[1])
        return loglik

    def estimate_loglik(self,data,**kwargs):
        loglik=0
        for w in tqdm(data):
            loglik+=np.log(self.importance_sample(w,**kwargs)[0])
        return loglik
    
    def estimate_grad(self,w,learn_bins,num_pi_b=1):
        """
        Approximate computation of the gradient of the log-likelihood given by 
        world w. Approximation based on random sample of pi,b combinations using the
        same sampling distribution as in importance sampling
        
        num_pi_b: number of random pi,b pairs based on which the approximation is computed
        
        Cf. -584-
        """
        UP=self.precompute_single_factors(w)
        BP=self.precompute_pair_factors(w)
        
        piweights=self.get_piweights(UP,BP)
        bweights = self.get_bweights(BP)
        
        nfac=math.factorial(w.n)
        
        # Generate random pi,b pairs:
        pi_b=[]
        for i in range(num_pi_b):
            ppi,pi=self.random_pi(w.n,piweights)
            pb,b=self.random_b_for_pi(pi,BP,UP,bweights)
            pi_b.append([pi,b,ppi*pb])
            
        #Initialize: 
        g1=list(np.zeros((self.granularity,self.signature.unaries[i]-1)) for i in self.attrange)
        if self.directed:
            g2 = np.zeros((self.granularity,self.granularity,self.signature.nb,2))
        else:
            g2 = np.zeros((self.granularity,self.granularity,self.signature.nb))

        p=0

        if learn_bins:
            gv=np.zeros(self.granularity+1)
        else:
            gv=None

        trace=[]   
        # Iterate over pi,b pairs:    
        for i in range(num_pi_b):
            
            g1n,g2n,pn=self.compute_grad(w,pi_b[i][0],pi_b[i][1])
            #print("pi: ",pi_b[i][0],"b: ",pi_b[i][1], " prob pi,b: ", pi_b[i][2], "prob w,pi,b: ", pn)
            if pn==0:
                print("zero prob. for pi", pi_b[0][0], " and b: ", pi_b[0][1])
                print("f1: ", self.f1[0])
                print("f2: ", self.f2)
          
            v,gvn=self.volume(pi_b[i][1],learn_bins)
            importanceweight=v/(pi_b[i][2])
            
           
            g1=list(g1[att] + g1n[att]*importanceweight  for att in self.attrange)
            trace.append(g1[0]/(i+1))

            #print("gradient b1,b1: ", (g2n)[1,1,0,0], (g2n)[1,1,0,1])            
            #print("weighted g2 gradient b1,b1: ", (g2n*importanceweight)[1,1,0,0], (g2n*importanceweight)[1,1,0,1])
            
            g2+=g2n*importanceweight
            if learn_bins:
                gv+=gvn*importanceweight
            p+=pn*importanceweight 
          
        p/=num_pi_b
        
        if p<=0:
            print("p = " ,p, " in estimate_grad")
            print("binbounds: ", self.binbounds)

        # Division by p for derivative of log and num_pi_b for normalization:
        divisor=p*num_pi_b
        
        if learn_bins:
            gv/=divisor
        g1=list(g1[att]/divisor  for att in self.attrange)
        g2/=divisor
        #return g1,g2,gv,np.log(p),trace 
        return g1,g2,gv,np.log(p)
        
    def estimate_grad_batch(self,batch,learn_bins,num_pi_b=1):
        gg1,gg2,ggv,p=self.estimate_grad(batch[0],learn_bins,num_pi_b)
        #print("batch first p: ", p)
        for i in range(1,len(batch)):
            gg1add,gg2add,ggvadd,padd=self.estimate_grad(batch[i],learn_bins,num_pi_b)
            gg1=list(gg1[att]+gg1add[att]  for att in self.attrange)
            gg2+=gg2add
            if learn_bins:
                ggv+=ggvadd
            p+=padd
           # print("batch next p: ",p)
        return gg1,gg2,ggv,p
        
        
    def compute_grad(self,w,pi,b): 
        """
        Computes the gradients of the f1,f2 functions for P(w,pi,b) (fixed pi and b). 
        Here no approximations and no log of probability/likelihood

        The gradient of f1 is only for the first m-1 independent parameters, i.e., the elements of
        gradf1 have one less column than the elements of f1 and countf1
        """
        gradf1= list(np.zeros((self.granularity,self.signature.unaries[i]-1)) for i in self.attrange)
        countf1= list(np.zeros((self.granularity,self.signature.unaries[i])) for i in self.attrange)
        
        if self.directed:
            gradf2 = np.zeros((self.granularity,self.granularity,self.signature.nb,2))
            countf2=np.zeros((self.granularity,self.granularity,self.signature.nb,2,2))
        else:
            gradf2 = np.zeros((self.granularity,self.granularity,self.signature.nb))
            countf2=np.zeros((self.granularity,self.granularity,self.signature.nb,2))
        
        

        # Populating the counts:

        # unary relations:
        for att in self.attrange:
            for i in range(w.n):
                if not w.unaries[pi[i],att]==-1:
                    countf1[att][b[i],w.unaries[pi[i],att]]+=1


        # binary relations:
        for ij in combinations(range(w.n),2): # iterating over unordered pairs of nodes
            for rel in range(self.signature.nb):
                if not self.directed:
                    ordered_pair=np.sort([pi[ij[0]],pi[ij[1]]])
                    if w.binaries[ordered_pair[0],ordered_pair[1],rel]==1:
                        countf2[b[ij[0]],b[ij[1]],rel,1]+=1
                    else:
                        countf2[b[ij[0]],b[ij[1]],rel,0]+=1
                if self.directed:        
                    if w.binaries[pi[ij[0]],pi[ij[1]],rel]==1:
                        countf2[b[ij[0]],b[ij[1]],rel,0,1]+=1
                    else:
                        countf2[b[ij[0]],b[ij[1]],rel,0,0]+=1
                    if w.binaries[pi[ij[1]],pi[ij[0]],rel]==1:
                        countf2[b[ij[0]],b[ij[1]],rel,1,1]+=1
                    else:
                        countf2[b[ij[0]],b[ij[1]],rel,1,0]+=1

   

        
        # Computing the product:
        prod=1
        for i in range(self.granularity):
            for att in self.attrange:
                for j in range(self.signature.unaries[att]):
                    prod*=self.f1[att][i,j]**countf1[att][i,j]
            for j in range(i,self.granularity):
                for rel in range(self.signature.nb):
                    if self.directed:
                        for d in (0,1):
                            prod*=self.f2[i,j,rel,d]**countf2[i,j,rel,d,1]*(1-self.f2[i,j,rel,d])**countf2[i,j,rel,d,0]
                    else:
                        prod*=self.f2[i,j,rel]**countf2[i,j,rel,1]*(1-self.f2[i,j,rel])**countf2[i,j,rel,0]
                        if prod==0:
                            print("got prod==0 with i: ",i, "j: ", j)
        # Computing the gradient:

        for i in range(self.granularity):
            for att in self.attrange:
                for j in range(self.signature.unaries[att]-1):
                    a=self.f1[att][i,j]
                    gradf1[att][i,j]=prod*(countf1[att][i,j]/a-countf1[att][i,-1]/self.f1[att][i,-1])
            for j in range(i,self.granularity):
                for rel in range(self.signature.nb):
                    if self.directed:
                        for d in (0,1):
                            a=self.f2[i,j,rel,d]
                            gradf2[i,j,rel,d]+=\
                            prod*(countf2[i,j,rel,d,1]*(1-a)-countf2[i,j,rel,d,0]*a)/(a*(1-a))
                    else:
                        a=self.f2[i,j,rel]
                        gradf2[i,j,rel]+=\
                        prod*(countf2[i,j,rel,1]*(1-a)-countf2[i,j,rel,0]*a)/(a*(1-a))

        return gradf1,gradf2,prod
        
    
            
    def sample_world(self,n,**kwargs):
        if "missing" in kwargs.keys():
            mp=kwargs['missing']
        else:
            mp=0.0
            
        pi = np.random.permutation(n)
        rng = np.random.default_rng()
        u_cont=rng.random(n)
        u_cont=np.sort(u_cont)
        b=np.zeros(n,dtype=int)
        for i in range(n):
            b[i]=self.get_binindx(u_cont[i])
        
#         #temporary hack (for enforcing equal sized communities in 2-community model):
#         b=np.hstack(((np.zeros(int(n/2),dtype=np.int32)),np.ones(int(n/2),dtype=np.int32)))
#         print("u: {}".format(b))
      
        w=World(self.signature,n) # All relations initialized with 0's throughout; 
        
        for att in self.attrange:
            for p in range(n):
                if rng.random()>mp:
                    j=np.random.choice(range(self.signature.unaries[att]),p=self.f1[att][b[p],:])
                    w.unaries[pi[p],att]=j
                else:
                    w.unaries[pi[p],att]=-1
        for br in range(self.signature.nb):
            for p in range(n):
                for q in range(p+1,n):
                    if self.directed:
                        if rng.random()>mp:
                            if rng.random() < self.f2[b[p],b[q],br,0]:
                                w.binaries[pi[p],pi[q],br]=1
                        else:
                            w.binaries[pi[p],pi[q],br]=-1
                        if rng.random()>mp:
                            if rng.random() < self.f2[b[p],b[q],br,1]:
                                w.binaries[pi[q],pi[p],br]=1
                        else:
                            w.binaries[pi[q],pi[p],br]=-1
                    else: #undirected
                        if rng.random()>mp:
                            if rng.random() < self.f2[b[p],b[q],br]:
                                w.binaries[np.min((pi[p],pi[q])),np.max((pi[p],pi[q])),br]=1
                        else:
                            w.binaries[np.min((pi[p],pi[q])),np.max((pi[p],pi[q])),br]=-1
        
        return w
                

    def splitbins(self,settings):
        # splits the widest bin into two
        bin_to_split=np.argmax(self.get_binlengths())
        print("to split: ", bin_to_split)
        self.granularity+=1
        newbounds=np.zeros(self.granularity+1)
        for i in range(bin_to_split+1):
            newbounds[i]=self.binbounds[i]
        newbounds[bin_to_split+1]=(self.binbounds[bin_to_split+1]+self.binbounds[bin_to_split])/2
        for i in range(bin_to_split+2,self.granularity+1):
            newbounds[i]=self.binbounds[i-1]
        self.binbounds=newbounds

        # Splitting f1:
        newf1=list(np.zeros((self.granularity,self.signature.unaries[i])) for i in self.attrange)
        # copying the bins before and after the split bin:
        for i in self.attrange:
            newf1[i][0:bin_to_split,:]=self.f1[i][0:bin_to_split,:]
            newf1[i][bin_to_split+1:,:]=self.f1[i][bin_to_split:,:]
        # initialzing the two new bins; the marginal distribution should remain unchanged
        for i in self.attrange:
            olddis=self.f1[i][bin_to_split,:]
            randdis=np.random.random(len(olddis))
            newf1[i][bin_to_split,:]=randdis*olddis
            newf1[i][bin_to_split+1,:]=(1-randdis)*olddis
        
        self.f1=newf1
         
        # Splitting f2:
        if self.directed:
                newf2 = np.zeros((self.granularity,self.granularity,self.signature.nb,2))
                newf2[0:bin_to_split,0:bin_to_split,:,:]=self.f2[0:bin_to_split,0:bin_to_split,:,:]
                newf2[bin_to_split+1:,bin_to_split+1:,:]=self.f2[bin_to_split+1:,bin_to_split+1:,:]
                newf2[0:bin_to_split,bin_to_split+1:,:]=self.f2[0:bin_to_split,bin_to_split+1:,:]
                # the former [bin_to_split,bin_to_split,:,:] elements have to be divided 3-ways:
                # [bin_to_split,bin_to_split,:,:],[bin_to_split,bin_to_split+1,:,:],[bin_to_split+1,bin_to_split+1,:,:]
                # where the non-diagonal one [bin_to_split,bin_to_split+1,:,:] has twice the weight (volume)
                randdis=np.random.random(3)
                randdis/=np.sum(randdis) # now have 3 probabilities adding to 1
                randdis[1]/=2 # decreasing the probability for the non-diagonal
                newf2[bin_to_split,bin_to_split,:,:]=randdis[0]*self.f2[bin_to_split,bin_to_split,:,:]
                newf2[bin_to_split,bin_to_split+1,:,:]=randdis[1]*self.f2[bin_to_split,bin_to_split+1,:,:]
                newf2[bin_to_split+1,bin_to_split+1,:,:]=randdis[2]*self.f2[bin_to_split+1,bin_to_split+1,:,:]
                # now splitting the remaining [bin_to_split, j ,:,:] entries
                for j in range(bin_to_split+2,self.granularity):
                    randdis=np.random.random()
                    newf2[bin_to_split,j,:,:]=randdis*self.f2[bin_to_split,j-1,:,:]
                    newf2[bin_to_split+1,j,:,:]=(1-randdis)*self.f2[bin_to_split,j-1,:,:]
                # and similarly for the [j,bin_to_split,:,:] entries
                for j in range(bin_to_split):
                    randdis=np.random.random()
                    newf2[j,bin_to_split,:,:]=randdis*self.f2[j,bin_to_split,:,:]
                    newf2[j,bin_to_split+1,:,:]=(1-randdis)*self.f2[j,bin_to_split,:,:]
                
        else:
                newf2 = np.zeros((self.granularity,self.granularity,self.signature.nb))
                newf2[0:bin_to_split,0:bin_to_split,:]=self.f2[0:bin_to_split,0:bin_to_split,:]
        
        self.f2=newf2
                
       
        return
        
    def learn(self,settings,data,**kwargs):
        """
        Arguments with prefixes ad_ are only for adam, prefixed with gr_ for greedy
        """
        learn_bins=settings['learn_bins']
        soft=settings['soft']
        num_pi_b=settings['num_pi_b']
        batchsize=settings['batchsize']
        numepochs=settings['numepochs']
        method=settings['method']
        with_trace=settings['with_trace']
        early_stop=settings['early_stop']
        
        assert method=="adam" or method=="greedy" or method=="RMSprop"

        if method=="adam":
            ad_alpha=settings['ad_alpha']
            ad_beta1=settings['ad_beta1']
            ad_beta2=settings['ad_beta2']
            ad_epsilon=settings['ad_epsilon']

        if method=="RMSprop":
            rms_alpha=settings['rms_alpha']
            rms_gamma=settings['rms_gamma']
            rms_lambda=settings['rms_lambda']
            rms_mu=settings['rms_mu']
            rms_centered=settings['rms_centered']
            rms_epsilon=settings['rms_epsilon']

        if method=="greedy":
            gr_lr=settings['gr_lr']
              
        
        assert method=="adam" or method=="greedy" or method=="RMSprop"

        
        trace=None
        if with_trace:
            trace={}
            trace['loglik']=[]
            trace['f1']=[]
            trace['f2']=[]
            
        #self.rand_init()
    
        numbatches=int(np.ceil(len(data)/batchsize))
            
        if method=="RMSprop":
            
           
            vf1=list(np.zeros((self.granularity,self.signature.unaries[i]-1))\
                     for i in self.attrange)
            vf2=np.zeros(shape=self.f2.shape)

            gavf1=list(np.zeros((self.granularity,self.signature.unaries[i]-1))\
                     for i in self.attrange)
            gavf2=np.zeros(shape=self.f2.shape)

            bf1=list(np.zeros((self.granularity,self.signature.unaries[i]-1))\
                     for i in self.attrange)
            bf2=np.zeros(shape=self.f2.shape)
            
            if learn_bins:
                bbb=np.zeros(len(self.binbounds))
                vbb=np.zeros(len(self.binbounds))
                gavbb=np.zeros(len(self.binbounds))
    

        if method=="adam":
            mf1=list(np.zeros((self.granularity,self.signature.unaries[i]-1))\
                     for i in self.attrange)
            mf2=np.zeros(shape=self.f2.shape)
            
            vf1=list(np.zeros((self.granularity,self.signature.unaries[i]-1))\
                     for i in self.attrange)
            vf2=np.zeros(shape=self.f2.shape)
            
            if learn_bins:
                vbb=np.zeros(len(self.binbounds))
                mbb=np.zeros(len(self.binbounds))       
            t=0
            
        terminate=False
        epochs=0
        
        batchidx=0
        low=int(batchidx*batchsize)
        high=int(np.min(((batchidx+1)*batchsize,len(data))))
        nextbatch=data[low:high]

        noimprov=0
        
        while not terminate:
            epochll=0
            for b in tqdm(range(numbatches)):          

                gf1,gf2,gbb,llbatch=self.estimate_grad_batch(nextbatch,learn_bins,num_pi_b)
                #print("grad f1: ", gf1[0],'\n')
                #print("grad f2 (b2,b2): ", gf2[1,1,0,0],gf2[1,1,0,1],'\n')
                
                epochll+=llbatch

                if method == "adam":
                    t+=1
                    mf1=list(ad_beta1*mf1[att]+(1-ad_beta1)*gf1[att] for att in self.attrange)
                    mf2=ad_beta1*mf2+(1-ad_beta1)*gf2
                    
                    vf1=list(ad_beta2*vf1[att]+(1-ad_beta2)*gf1[att]**2 for att in self.attrange)
                    vf2=ad_beta2*vf2+(1-ad_beta2)*gf2**2
                    
                    hatmf1=list(mf1[att]/(1-ad_beta1**t) for att in self.attrange)
                    hatmf2=mf2/(1-ad_beta1**t)

                    hatvf1=list(vf1[att]/(1-ad_beta2**t) for att in self.attrange)
                    hatvf2=vf2/(1-ad_beta2**t)

                    for att in self.attrange:
                        self.f1[att][:,0:-1]=self.f1[att][:,0:-1]+ad_alpha*hatmf1[att]/(np.sqrt(hatvf1[att])+ad_epsilon)

                    self.f2=self.f2+ad_alpha*hatmf2/(np.sqrt(hatvf2)+ad_epsilon)
                    
                    if learn_bins:
                        mbb=ad_beta1*mbb+(1-ad_beta1)*gbb
                        vbb=ad_beta2*vbb+(1-ad_beta2)*gbb**2
                        hatmbb=mbb/(1-ad_beta1**t)
                        hatvbb=vbb/(1-ad_beta2**t)
                        self.binbounds+=ad_alpha*hatmbb/(np.sqrt(hatvbb)+ad_epsilon)

                    #print("after adam: ", self.f1[0], '\n')
                    #print("after adam  f2 (b2,b2): ", self.f2[1,1,0,0],self.f2[1,1,0,1],'\n')
                
                if method == "RMSprop":
                    if not rms_lambda ==0:
                        gf1=list(gf1[att]+rms_lambda*self.f1[att][:,0:-1] for att in self.attrange)
                        gf2=gf2+rms_lambda*self.f2

                    vf1=list(rms_alpha*vf1[att]+(1-rms_alpha)*gf1[att]**2 for att in self.attrange)
                    vf2=rms_alpha*vf2+(1-rms_alpha)*gf2**2

                    vtildef1=list(vf1[att].copy() for att in self.attrange)
                    vtildef2=vf2.copy()

                    if rms_centered:
                         gavf1=list(rms_alpha*gavf1[att]+(1-rms_alpha)*gf1[att] for att in self.attrange)
                         gavf2=rms_alpha*gavf2+(1-rms_alpha)*gf2
                         vtildef1=list(vtildef1[att]-gavf1[att]**2 for att in self.attrange)
                         vtildef2=vtildef2-gavf2**2

                    if rms_mu>0:
                        bf1=list(rms_mu*bf1[att]+gf1[att]/(np.sqrt(vtildef1[att])+rms_epsilon) for att in self.attrange)
                        bf2=rms_mu*bf2+gf2/(np.sqrt(vtildef2)+rms_epsilon)
                        self.f1=list(self.f1[att]+rms_gamma*bf1[att] for att in self.attrange)
                        self.f2=self.f2+rms_gamma*bf2
                    else:
                        self.f1=list(self.f1[att]+rms_gamma*gf1[att]/(np.sqrt(vtildef1[att])+rms_epsilon) for att in self.attrange)
                        self.f2=self.f2+rms_gamma*gf2/(np.sqrt(vtildef2)+rms_epsilon)
                        
                    if learn_bins:
                        gbb=gbb+rms_lambda*self.binbounds
                        vbb=rms_alpha*vbb+(1-rms_alpha)*gbb**2
                        vtildebb=vbb.copy()
                        if rms_centered:
                            gavbb=rms_alpha*gavbb+(1-rms_alpha)*gbb
                            vtildebb=vtildebb-gavbb**2
                        if rms_mu>0:
                            bbb=rms_mu*bbb+gbb/(np.sqrt(vtildebb)+rms_epsilon)
                            self.binbounds=self.binbounds+rms_gamma*bbb
                        else:
                            self.binbounds=self.binbounds+rms_gamma*gbb/(np.sqrt(vtildebb)+rms_epsilon)
                    
                if method == "greedy":
                    for att in self.attrange:
                        gf1[att]=gf1[att]/np.linalg.norm(gf1[att])
                        self.f1[att][:,0:-1]+=gr_lr*gf1[att]
                        
                    gf2=gf2/np.linalg.norm(gf2)
                    self.f2+=gr_lr*gf2

                    if learn_bins:
                        gbb=gbb/np.linalg.norm(gbb)
                        binupdate=gr_lr*gbb

                # Smoothing the f1 values and inserting last component:
                for att in self.attrange:
                    # impose 'soft' as minimal value:
                    self.f1[att][:,0:-1]=np.maximum(self.f1[att][:,0:-1],soft)
                    #print("after soft-min:", self.f1[att])
                    # impose 1-soft as maximal sum of first m-1 values:
                    f1sum=np.sum(self.f1[att][:,0:-1],axis=1)
                    normfac=np.minimum(f1sum,1-soft)/f1sum
                    self.f1[att][:,0:-1]=self.f1[att][:,0:-1]*normfac[:,np.newaxis]
                    #print("after normalizing first m-1", self.f1[att])
                    self.f1[att][:,-1]=1-np.sum(self.f1[att][:,0:-1],axis=1)
                    #print("after entering last col.", self.f1[att])        

                # Smoothing the f2 values:
                self.f2=np.minimum(self.f2,1-soft)
                self.f2=np.maximum(self.f2,soft)

                if learn_bins:
                # Constrain binbounds
                    # Sort, in case some bounds have crossed each other:
                    self.binbounds=np.sort(self.binbounds)
                    # Bring back to [0,1]
                    self.binbounds=(self.binbounds-self.binbounds[0])/(self.binbounds[-1]-self.binbounds[0])

                
                batchidx=(batchidx+1)%numbatches
                low=int(batchidx*batchsize)
                high=int(np.min(((batchidx+1)*batchsize,len(data))))
                
                nextbatch=data[low:high]
            
            epochs+=1
            
            #if method == "adam":
            #    ad_alpha=0.9*ad_alpha
                
            if epochs==1:
                bestmodel=self.copy()
                bestloglik=epochll
            else:
                if epochll>bestloglik:
                    bestloglik=epochll
                    bestmodel=self.copy()
                    noimprov=0
                else:
                    noimprov+=1
            print("Epoch",  epochs,  "log-lik: ", epochll, "early_stop: " ,early_stop, "noimprov: ", noimprov)
            
            if with_trace:
                trace['loglik'].append(epochll)
                trace['f1'].append(self.f1[0].copy())
                trace['f2'].append(self.f2.copy())
                
            terminate=(epochs==numepochs) or (noimprov==early_stop)
        
        return bestmodel,bestloglik,trace
   
        
