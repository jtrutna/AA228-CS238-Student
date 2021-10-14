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


def _bayesian_score_for_ij(M_ij, a_ij):
    p = sum()

    # TODO: 
def compute(infile, outfile):
    df = pd.read_csv(infile, index_col=False)
    n = len(df.columns)
    r_i = df.nunique()

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    print(parents(2, G))

    print(prior(n, G, r_i))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=argparse.FileType('r'), default='data/small.csv')
    parser.add_argument('--outfile', type=argparse.FileType('w'), default='out.gph')
    args = parser.parse_args()

    compute(infile=args.infile, outfile=args.outfile)


if __name__ == '__main__':
    main()
