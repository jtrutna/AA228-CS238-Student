import argparse
import sys

import networkx as nx
import pandas as pd


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def prior(G, r_i):
    n = G.number_of_nodes()


def compute(infile, outfile):
    df = pd.read_csv(infile, index_col=False)
    # Get nodes in a form that DiGraph can easily digest (with labels)
    # nodes = {0: {'label': 'age', 'm': 3}, 1: {'label': 'portembarked', 'm': 3}, ...
    nodes = df.nunique().to_frame('m').reset_index().rename(columns={'index': 'label'})
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    print(G.nodes[1])
    r_i = df.nunique()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=argparse.FileType('r'), default='data/small.csv')
    parser.add_argument('--outfile', type=argparse.FileType('w'), default='out.gph')
    args = parser.parse_args()

    compute(infile=args.infile, outfile=args.outfile)


if __name__ == '__main__':
    main()
