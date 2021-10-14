import argparse
from os.path import basename
from random import randrange
import sys

import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import loggamma


def write_gph(dag, idx2names, outfile):
    for edge in dag.edges():
        outfile.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def parents(i, G):
    # TODO: Check that i < G.nodes.length
    # G.adjacency => {0: {}, 1: {2: {}}} where 1=>2
    return [k for k, v in G.adj.items() if i in v]


def parental_instantiations(n, G, r_i):
    q_i = np.ones(n, dtype=np.uint64)
    # G.adjacency => {0: {}, 1: {2: {}}} where 1=>2
    for k, children in G.adjacency():
        for child in children.keys():
            q_i[child] *= r_i[k]
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
            if len(ps) > 0:
                parent_values = [row[j] for j in parents(i, G)]
                parent_rs = [r_i[j] for j in parents(i, G)]
                # parential instantiations of X_i can be thought of a w-dimensional matrix where w
                # is the number of parents of X_i and the d-dimension can take r_i values.
                # q_i is this flattened, so we calculate where [i_1, i_2, ..., i_h] is in flattened
                j = np.ravel_multi_index(parent_values, parent_rs)
            m_ijk[i][j, k] += 1
    return m_ijk


def bayescore_for_i(m_jk, a_jk):
    score = np.sum(loggamma(np.sum(a_jk, axis=1)))
    score -= np.sum(loggamma(np.sum(a_jk, axis=1) + np.sum(m_jk, axis=1)))
    score += np.sum(loggamma(a_jk + m_jk))
    score -= np.sum(loggamma(a_jk))
    return score


def bayscore(n, G, r_i, D):
    a_ijk = prior(n, G, r_i)
    # TODO: Check for order of columns
    m_ijk = evidence_counts_by_ijk(n, G, r_i, D)
    return sum(bayescore_for_i(m_ijk[i], a_ijk[i]) for i in range(n))


def _random_neighbor(n, orig_G, r_i, D):
    G = orig_G.copy()
    i = randrange(n)
    j = randrange(n)
    while j == i:
        j = randrange(n)
    if G.has_edge(i, j):
        G.remove_edge(i, j)
        if bool(randrange(1)):
            G.add_edge(j, i)
    else:
        G.add_edge(i, j)
    return G


def _has_cycle(G):
    try:
        nx.algorithms.cycles.find_cycle(G)
        return True
    except nx.exception.NetworkXNoCycle:  # :|
        return False


def fit(n, G, r_i, D, iterations=10):
    current_score = bayscore(n, G, r_i, D)
    for _ in range(iterations):
        print(f"{G.edges()}:{current_score}")
        candidate_G = _random_neighbor(n, G, r_i, D)
        if _has_cycle(candidate_G):
            continue
        score = bayscore(n, candidate_G, r_i, D)
        if score > current_score:
            G, current_score = candidate_G, score

    return G


def compute(infile, outfile):
    D_raw = pd.read_csv(infile, index_col=False, dtype="category")
    # XXX:For simpler internals and since we're not actually outputting field values
    #     we replace each field with an index k into the possible values for that column.
    n = len(D_raw.columns)
    D = D_raw.apply(lambda x: x.cat.codes)
    r_i = D.nunique()
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    Gp = fit(n, G, r_i, D)
    print(Gp)

    write_gph(Gp, D.columns, outfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=argparse.FileType('r'), default='data/toy.csv')
    parser.add_argument('--output')
    args = parser.parse_args()


    if args.output is None:
        output = basename(args.input.name).replace('.csv', '') + '.gph'

    with open(output, 'w') as f:
        compute(infile=args.input, outfile=f)


if __name__ == '__main__':
    main()
