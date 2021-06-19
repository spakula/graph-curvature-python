import argparse, logging

import networkx as nx
import numpy as np
import scipy.sparse as spsp
import scipy.optimize as spop
import matplotlib.pyplot as plt

#################################################################################
# helpers

# returns the measure that shows up on Ollivier-Ricci curvature expression
#  (with idleness p)
# return value is a dict, "vertex": weight (could be zero)
def OR_sphere_measure(G, vtx, p=0):
    deg = nx.degree(G,vtx)
    return {v: (1-p)/deg for v in nx.neighbors(G,vtx)} | { vtx: p }


# arrange parameters for the linear programming problem that computes
# the Wasserstein distance between two measures m1, m2 on the vertices
# of a graph G
# from https://arxiv.org/pdf/1712.03033.pdf page 10
def Wasserstein_get_linprog_params(metric, m1, m2):
    # filter zeroes
    m1f = { v:w for v,w in m1.items() if w != 0 }
    m2f = { v:w for v,w in m2.items() if w != 0 }
    # supports
    m1_supp = sorted(m1f.keys())
    m2_supp = sorted(m2f.keys())
    en = len(m1_supp)
    em = len(m2_supp)

    # minimise dot product with m
    m = [ - m1f[v] for v in m1_supp ] + [ - m2f[v] for v in m2_supp ]

    # inequality matrix
    A = spsp.hstack([
        spsp.vstack([spsp.coo_matrix(([1]*em,(range(em),[i]*em)),shape=(em,en)) for i in range(en)]),
        spsp.vstack([spsp.eye(em,dtype=np.int64)]*en)
        ])
    # inequality RHS vector
    c = [ metric[x][y] for x in m1_supp for y in m2_supp ]

    # find all vertices that are in both supports, and translate to an equation matrix
    m_both = [ v for v in m1_supp if v in m2_supp ]
    if m_both == []:
        Aeq = None
    else:
        rows = [ i for i in range(len(m_both)) for _ in range(2) ]
        cols = [f(v) for v in m_both for f in 
            (lambda t: m1_supp.index(t), lambda t: en+m2_supp.index(t)) ]
        data = [1,1] * len(m_both)
        Aeq = spsp.coo_matrix((data,(rows,cols)),shape=(len(m_both),en+em))

    return (m,A.todense(),c,(Aeq.todense() if Aeq is not None else None))


# compute the Wasserstein W₁ distance between measures m1 and m2
# metric is the path metric on the graph
# (Wasserstein distance only depends on the distances between the
#  points in the supports of m1 and m2)
def Wasserstein(metric, m1, m2):
    (m,A,c,Aeq) = Wasserstein_get_linprog_params(metric, m1, m2)
    # print(m,A,c,Aeq)
    res = spop.linprog(m, A_ub=A, b_ub=c, A_eq=Aeq, b_eq=(None if Aeq is None else [0]*(Aeq.shape[0])), bounds=(None,None))
    # print(res.success)
    return -res.fun


# compute the Ollivier-Ricci curvature between all pairs of
# (different) vertices
# (not a sparse matrix)
# (diagonal entries are left to be 0)
def OR_get_full_curvature_matrix(G, p=0):
    metric = dict(nx.all_pairs_shortest_path_length(G))

    vtxs = sorted(G.nodes)
    n = len(vtxs)

    cv = np.zeros((n,n))
    for i in range(len(vtxs)):
        for j in range(i+1,len(vtxs)):
            cv[i,j] = 1 - (Wasserstein(metric, OR_sphere_measure(G,vtxs[i],p), OR_sphere_measure(G,vtxs[j],p)))/metric[vtxs[i]][vtxs[j]]
            cv[j,i] = cv[i,j]

    return cv


# compute the Ollivier-Ricci curvature between all pairs of
# adjacent vertices
# (returns a sparse matrix)
def OR_get_edge_curvature_matrix(G, p=0):
    metric = dict(nx.all_pairs_shortest_path_length(G))

    row_ind = []
    col_ind = []
    data = []
    for (i,j) in G.edges:
        row_ind += [i,j]
        col_ind += [j,i]
        data.append(1 - (Wasserstein(metric, OR_sphere_measure(G,i,p), OR_sphere_measure(G,j,p)))/metric[i][j])
        data.append(data[-1])

    return spsp.csr_matrix((data,(row_ind,col_ind)),shape=(len(G.nodes),len(G.nodes)))


# check if the matrix A is positive (semi-)definite
# real matrices only
# (attempts cholesky, which bails when the matrix is not >=0)
# (for semidefiniteness, we add a small multiple of Id and check that)
def is_pos_def(A, semi=True):
    # A is >=  iff  A+A^T >= 0
    M = A + A.T
    try:
        # regularize if want _semi_definiteness
        np.linalg.cholesky(
            (M + np.eye(M.shape[0]) * 1e-14) if semi else M
            )
        return True
    except np.linalg.LinAlgError:
        return False


# check if the matrix A is conditionally negative definite
# uses the trick from math overflow: rows of "p" are a basis of
# the orthogonal complement of the subspace of constant functions;
# so p @ A @ p.T is a matrix of the form restricted to that complement
# and then we test _minus that_ for positive (semi-)definiteness
def is_cond_neg_def(A, semi=True):
    r = A.shape[0] - 1
    p = np.hstack([np.array([[1]]*r),-np.eye(r,dtype=np.int64)])
    return is_pos_def((-1) * p @ A @ p.T, semi)


#################################################################################
def main(args, loglevel):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)

    ######### MESS HERE #########

    G = nx.circular_ladder_graph(3)
    # G = nx.petersen_graph()
    # G = nx.ladder_graph(4)
    # G = nx.star_graph(3)
    # G = nx.wheel_graph(5)
    # G = nx.complete_graph(5)
    # G = nx.cycle_graph(6)
    # G = nx.lollipop_graph(4,3)
    # curv = OR_get_full_curvature_matrix(G)
    curv = OR_get_edge_curvature_matrix(G).todense()
    curv = np.around(curv,5)

    print(curv)
    # print(np.rint(3*curv))
    # print(np.linalg.eigh(curv))

    r = curv.shape[0] - 1
    p = np.hstack([np.array([[1]]*r),-np.eye(r,dtype=np.int64)])
    restr_curv = p @ curv @ p.T  # 3* and rint just to get a nicer matrix
    print("is curvature matrix conditionally negative definite: ", is_pos_def((-1)*restr_curv))
    # print(restr_curv)

    # print(np.linalg.eigh(restr_curv))
    print("curv eigvals: ", sorted(np.linalg.eigvals(curv)))
    print("restricted curv eigvals: ", sorted(np.linalg.eigvals(restr_curv)))

    # print(is_cond_neg_def(-nx.laplacian_matrix(G),semi=False))

    # metric = dict(nx.all_pairs_shortest_path_length(G))
    # ma = OR_sphere_measure(G,0)
    # mb = OR_sphere_measure(G,1)
    # print(ma, mb)
    # print(Wasserstein(metric, ma, mb))

    ## draw the graph
    # plt.subplot(121)
    # nx.draw(G,with_labels=True)
    # plt.show()






#################################################################################
#################################################################################
#################################################################################
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Play around with graph curvature.",
                                    epilog = "As an alternative to the commandline, params can be placed in a file, one per line, and specified on the commandline like '%(prog)s @params.conf'.",
                                    fromfile_prefix_chars = '@' )
    # TODO Specify your real parameters here.
    parser.add_argument("-v",
                        "--verbose",
                        help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    main(args, loglevel)