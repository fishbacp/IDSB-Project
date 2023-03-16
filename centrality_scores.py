import numpy as np
import networkx as nx
import random
from scipy import sparse
import numpy.linalg as linalg

def betweenness_score(A):
    G=nx.from_numpy_matrix(A)
    BC=nx.betweenness_centrality(G,normalized=True,weight='weight')
    BC=list(BC.values())
    return BC

def reverse_page_rank_score(A):
# Large reverse PageRank values suggest nodes that can reach many nodes in the graph
# See chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1407.5107.pdf
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    Scores=nx.pagerank_numpy(G,weight='weight')
    Scores=list(Scores.values())
    return Scores

def generalized_degree_centrality(S,alpha):
    #np.fill_diagonal(S,0)
    s=S.sum(axis=0)
    Adj=np.where(S>0,1,0)
    np.fill_diagonal(Adj,0)

    print(np.round(S,3))
    print(np.round(Adj,3))
    
    k=Adj.sum(axis=0)
    GDC=[k[i]**(1-alpha)*(s[i]-1)**alpha for i in range(len(k))]
    return GDC


'''
def eigenvector_score(A):
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    #EC=nx.eigenvector_centrality(G,max_iter=1000,weight='weight')
    EC = nx.eigenvector_centrality_numpy(G,weight='weight',max_iter=10000,tol=1e-3)
    EC=list(EC.values())
    return EC
'''

def eigenvector_score(A):
    eigenValues, eigenVectors = linalg.eig(A)
    print('SWDTF')
    print(np.round(A,3))
    print('eigen info')
    print(np.round(eigenValues,3))
    print(np.round(eigenVectors,3))
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    #eigenVectors = eigenVectors[:,idx]
    EC=abs(eigenVectors[:,idx[0]])
    EC=np.real(1/np.amax(EC)*EC)
    print('EC')
    print(np.round(EC,3))
    return EC

def eigenvector_score_without_graph(X, alpha=0.85, max_iter=1000, tol=1e-5):
    X= sparse.csr_matrix(X)
    n = X.shape[0]
    X = X.copy()
    incoming_counts = np.asarray(X.sum(axis=1)).ravel()

    for i in incoming_counts.nonzero()[0]:
        X.data[X.indptr[i] : X.indptr[i + 1]] *= 1.0 / incoming_counts[i]
    dangle = np.asarray(np.where(np.isclose(X.sum(axis=1), 0), 1.0 / n, 0)).ravel()

    scores = np.full(n, 1.0 / n, dtype=np.float32)  # initial guess
    for i in range(max_iter):
        prev_scores = scores
        scores = (
            alpha * (scores * X + np.dot(dangle, prev_scores))
            + (1 - alpha) * prev_scores.sum() / n
        )
        # check convergence: normalized l_inf norm
        scores_max = np.abs(scores).max()
        if scores_max == 0.0:
            scores_max = 1.0
        err = np.abs(scores - prev_scores).max() / scores_max
        if err < n * tol:
            return scores


    return scores



# X=np.random.rand(8,8)

# Xs = sparse.csr_matrix(X)
# scores=eigenvector_score(Xs, alpha=0.85, max_iter=100, tol=1e-10)
# s=sum(scores)
# scores_normalized=1/s*scores
# print(np.round(scores,4))


# scores=eigenvector_score_with_graph(X)  # Requires construction network first.
# s=sum(scores)
# scores_normalized=[1/s*i for i in scores]
# print(np.round(scores_normalized,4))





'''
A=np.random.random((5,5))
print(np.round(A+A.T,2))

BC=betweenness_score(A+A.T)
print(BC)
'''
