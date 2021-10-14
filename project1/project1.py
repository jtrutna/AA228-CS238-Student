import argparse
import sys

import networkx as nx
import pandas as pd


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def q_i(n, G, r_i):
    q_i = dict((i, 1) for i in range(n))
    # G.adjacency => {0: {}, 1: {2: {}}} where 1=>2
    for k, children in G.adjacency():
        for child in children.keys():
            q_i[child] *= r_i[child]
    print(q_i)

def prior(n, G, r_i):
    assert G.number_of_nodes() == n


def compute(infile, outfile):
    df = pd.read_csv(infile, index_col=False)
    n = len(df.columns)
    r_i = df.nunique()

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    q_i(n, G, r_i)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=argparse.FileType('r'), default='data/small.csv')
    parser.add_argument('--outfile', type=argparse.FileType('w'), default='out.gph')
    args = parser.parse_args()

    compute(infile=args.infile, outfile=args.outfile)


if __name__ == '__main__':
    main()
