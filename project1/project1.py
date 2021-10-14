import argparse
import sys

import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import loggamma


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def parents(i, G):
    # G.adjacency => {0: {}, 1: {2: {}}} where 1=>2
    return [k for k, v in G.adj.items() if i in v]


def parental_instantiations(n, G, r_i):
    q_i = np.ones(n, dtype=np.uint64)
    # G.adjacency => {0: {}, 1: {2: {}}} where 1=>2
    for k, children in G.adjacency():
        for child in children.keys():
            q_i[child] *= r_i[child]
    return q_i


def prior(n, G, r_i):
    q_i = parental_instantiations(n, G, r_i)
    return [np.ones((q_i[i], r_i[i]), dtype=np.uint64) for i in range(n)]


def evidence_counts_by_ijk(n, G, r_i, D):
    # Number of times each node, parential instantiation, and node value appears in D
    q_i = parental_instantiations(n, G, r_i)
    m_ijk = [np.zeros([q_i[i], r_i[i]]) for i in range(n)]
    for row in D.itertuples(index=False):
        for i in range(n):
            k = row[i]
            j = 0
            ps = parents(i, G)
            if len(ps) > 1:
                parent_values = [row[i] for i in parents(i, G)]
                parent_rs = [r_i[i] for i in parents(i, G)]
                # parential instantiations of X_i can be thought of a w-dimensional matrix where w
                # is the number of parents of X_i and the d-dimension can take r_i values.
                # q_i is this flattened, so we calculate where [i_1, i_2, ..., i_h] is in flattened
                j = np.ravel_multi_index(parent_values, parent_rs)
            m_ijk[i][j, k] += 1
    return m_ijk


    # TODO: 
def compute(infile, outfile):
    D_raw = pd.read_csv(infile, index_col=False, dtype="category")
    n = len(D_raw.columns)

    # XXX:For simpler internals, we replace each value with an index k for the
    # k-th possible value for that column
    variable_denormalize = {i: D_raw.iloc[:,i].cat.categories for i in range(n)}
    D = D_raw.apply(lambda x: x.cat.codes)
    
    r_i = D.nunique()

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edge(1,0)

    a_ijk = prior(n, G, r_i)
    # TODO: Check for order of columns
    m_ijk = evidence_counts_by_ijk(n, G, r_i, D)

    score = sum(bayescore_for_i(m_ijk[i], a_ijk[i]) for i in range(n))
    return score


def bayescore_for_i(m_jk, a_jk):
    score = np.sum(loggamma(np.sum(a_jk, axis=1)))
    score -= np.sum(loggamma(np.sum(a_jk, axis=1) + np.sum(m_jk, axis=1)))
    score += np.sum(loggamma(a_jk + m_jk))
    score -= np.sum(loggamma(a_jk))
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=argparse.FileType('r'), default='data/small.csv')
    parser.add_argument('--outfile', type=argparse.FileType('w'), default='out.gph')
    args = parser.parse_args()

    print(compute(infile=args.infile, outfile=args.outfile))


if __name__ == '__main__':
    main()
