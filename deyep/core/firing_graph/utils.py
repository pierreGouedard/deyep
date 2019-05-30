# Global imports
import numpy as np
from scipy.sparse import lil_matrix

from deyep.core.firing_graph import vertex
from deyep.core.tools.basis.comon import Basis


def mat_from_tuples(l_edges, n_iv, n_cv, n_ov, weights=None):
    """
    Take list of tuple (str x, str y), dimension of network and weights and set matrices of network
    :param l_edges: [(str x, str y)] sontains name of vertex class  class
    :param n_iv: int number of input vertex
    :param n_cv: int number of core vertex
    :param n_ov: int number of core vertex
    :param weights: either not set of int weights of every edges or list of weight (size equal to size of l_edges
    :return: 3 sparse matrices of the network
    """
    # Init matrices
    sax_in = lil_matrix(np.zeros([n_iv, n_cv]))
    sax_core = lil_matrix(np.zeros([n_cv, n_cv]))
    sax_out = lil_matrix(np.zeros([n_cv, n_ov]))

    i = 0
    for (_n, n_) in l_edges:
        if 'input' in _n:
            if weights is None:
                v = np.random.randint(1, 100)
            elif isinstance(weights, int):
                v = weights
            elif isinstance(weights, list):
                try:
                    v = weights[i]
                except IndexError:
                    print "Dimension of weights does not match dimension of list of edges"
            else:
                raise ValueError("Value of the weights not correct")

            sax_in[int(_n.split('_')[1]), int(n_.split('_')[1])] = v

        elif 'core' in _n:
            if 'core' in n_:
                if weights is None:
                    v = np.random.randint(1, 100)
                elif isinstance(weights, int):
                    v = weights
                elif isinstance(weights, list):
                    try:
                        v = weights[i]
                    except IndexError:
                        print "Dimension of weights does not match dimension of list of edges"
                else:
                    raise ValueError("Value of the weights not correct")

                sax_core[int(_n.split('_')[1]), int(n_.split('_')[1])] = v

            elif 'output' in n_:
                if weights is None:
                    v = np.random.randint(1, 100)
                elif isinstance(weights, int):
                    v = weights
                elif isinstance(weights, list):
                    try:
                        v = weights[i]
                    except IndexError:
                        print "Dimension of weights does not match dimension of list of edges"
                else:
                    raise ValueError("Value of the weights not correct")

                sax_out[int(_n.split('_')[1]), int(n_.split('_')[1])] = v

        i += 1

    return sax_in.tocsc(), sax_core.tocsc(), sax_out.tocsc()


def vertices_from_mat(sax_core, sax_in, sax_out, capacity, ax_levels):
    """
    Create the list of vertices composing a firing graph from sparse matrices that describe its topology

    :param sax_core: scipy.sparse matrix of core vertices
    :param sax_in: scipy.sparse matrix of input vertices
    :param sax_out: scipy.sparse matrix of output vertices
    :param capacity: int number of freqs by nodes
    :param ax_levels: array of minimum level of vertices of firing graph
    :return:
    """

    # Get dict of nodes
    d_inputs = set_vertices_from_mat(sax_in, 'input')
    d_outputs = set_vertices_from_mat(sax_out, 'output')
    d_cores = set_vertices_from_mat(sax_core, 'core')

    # distribute frequency among input nodes,
    d_inputs, set_freqs, d_forward_freqs = set_frequencies(d_inputs, {0}, 1, {})

    # distribute frequencies among core nodes
    d_cores, set_freqs_, d_forward_freqs = set_frequencies(
        d_cores, {0}, capacity, d_forward_freqs, offset=len(set_freqs)
    )
    n_freq = ((max(set_freqs_) + 1) * capacity) + len(set_freqs)

    # Finally, build nodes
    l_inputs = [
        vertex.InputVertex(
            k_, 'input', Basis(min(v_['freqs']), n_freq, {}, 1),
            zip(v_['children_name'], v_['children_weight'])
        ) for k_, v_ in sorted(d_inputs.items(), key=lambda (k, v): k)
        ]
    l_cores = [
        vertex.CoreVertex(
            k_, 'core',
            Basis(min(v_['freqs']), n_freq, d_forward_freqs.get('core_{}'.format(k_), []), capacity),
            zip(v_['children_name'], v_['children_weight']), ax_levels[k_]
        ) for k_, v_ in sorted(d_cores.items(), key=lambda (k, v): k)
        ]

    l_outputs = [vertex.OutputVertex(k_, 'output', zip(v_['parents_name'], v_['parents_weight']))
                 for k_, v_ in sorted(d_outputs.items(), key=lambda (k, v): k)]

    return l_inputs, l_outputs, l_cores, n_freq


def set_vertices_from_mat(mat, key):
    """
    build a dictionnary version of vertex
    :param mat: scipy.sparse matrice of the graph
    :param key: str in 'input', 'core' or 'output
    :return:
    """
    if key == 'input':
        d_vertices = {i: {'children_name': [], 'children_weight': [], 'freqs': []} for i in range(mat.shape[0])}
        for i, j in zip(*mat.nonzero()):
            d_vertices[i]['children_name'] += ['core_{}'.format(j)]
            d_vertices[i]['children_weight'] += [mat[i, j]]

    elif key == 'output':
        d_vertices = {i: {'parents_name': [], 'parents_weight': [], 'freqs': []} for i in range(mat.shape[1])}
        for i, j in zip(*mat.nonzero()):
            d_vertices[j]['parents_name'] += ['core_{}'.format(i)]
            d_vertices[j]['parents_weight'] += [mat[i, j]]

    elif key == 'core':
        d_vertices = {i: {'children_name': [], 'children_weight': [], 'freqs': []} for i in range(mat.shape[0])}
        for i, j in zip(*mat.nonzero()):
            d_vertices[i]['children_name'] += ['core_{}'.format(j)]
            d_vertices[i]['children_weight'] += [mat[i, j]]
    else:
        raise ValueError('Choose key between \'input\', \'output\' or \'core\'')

    return d_vertices.copy()


def set_frequencies(d_vertices, freqs, capacity, d_forward_freqs, offset=0):
    """
    Distribute and add minimal frequency to vertex of the firing graph
    :param d_vertices: dictionary of vertex that host the frequencies
    :param freqs: set of already used frequencies
    :param capacity: Number of frequency by vertex
    :param d_forward_freqs: list of frequency indices that vertex may received
    :param offset: max frequency index already allocated
    :return:
    """
    for k, v in d_vertices.items():
        if len(v['children_name']) > 0:
            # Look for sibling
            freqs_ = freqs.difference(get_occupied_freqs(k, v['children_name'], d_vertices))

            if len(freqs_) > 0:
                # Distribute available freqs
                d_vertices[k]['freqs'] = [list(freqs_)[0]]

            else:
                # Add new frequencies
                d_vertices[k]['freqs'] = [max(freqs) + 1]
                freqs = freqs.union([max(freqs) + 1])
        else:
            d_vertices[k]['freqs'] = [list(freqs)[0]]

        if d_forward_freqs is not None:
            for child in v['children_name']:
                d_forward_freqs[child] = d_forward_freqs.get(child, []) + [offset + (d_vertices[k]['freqs'][0] * capacity)]

    for k, v in d_vertices.items():
        v['freqs'] = [offset + (min(v['freqs']) * capacity)]

    return d_vertices, freqs, d_forward_freqs


def get_occupied_freqs(k, l_children, d_vertices):
    """
    Get occupied freqs
    :param k: int key of the vertex
    :param l_children: list of key of vertices
    :param d_vertices: list of all vertices
    :return: set of int that identify frequencies that assigned to vetices identified by list of key n l_children
    """
    occupied_freqs = []
    for child in l_children:
        for k_, v_ in d_vertices.items():
            if child in v_['children_name'] and k != k_ and len(v_['freqs']) > 0:
                occupied_freqs += v_['freqs']

    return set(occupied_freqs)