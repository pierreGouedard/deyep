# Global imports
from scipy.sparse import csc_matrix
import numpy as np

# local import
import settings
from deyep.core import nodes
from deyep.core.tools.frequencies import FrequencyStack


def mat_from_tuples(l_edges, n_i, n_rn, n_o, weights='random'):

    # Init matrices
    mat_in = csc_matrix(np.zeros([n_i, n_rn]))
    mat_net = csc_matrix(np.zeros([n_rn, n_rn]))
    mat_out = csc_matrix(np.zeros([n_rn, n_o]))

    i = 0
    for (_n, n_) in l_edges:
        if 'input' in _n:
            if weights == 'random':
                v = np.random.randint(1, 100)
            else:
                v = weights[i]

            mat_in[int(_n.split('_')[1]), int(n_.split('_')[1])] = v

        elif 'network' in _n:
            if 'network' in n_:
                if weights == 'random':
                    v = np.random.randint(1, 100)
                else:
                    v = weights[i]

                mat_net[int(_n.split('_')[1]), int(n_.split('_')[1])] = v

            elif 'output' in n_:
                if weights == 'random':
                    v = np.random.randint(1, 100)
                else:
                    v = weights[i]

                mat_out[int(_n.split('_')[1]), int(n_.split('_')[1])] = v

        i += 1

    return mat_in, mat_net, mat_out


def mat_from_nodes(l_nodes):
    raise NotImplementedError


def nodes_from_mat(mat_net, mat_in, mat_out, capacity, l0=10, tau=5):

    # Get dict of nodes
    d_inputs, d_net_ = set_nodes_from_mat(mat_in, 'input')
    d_outputs, d_net_ = set_nodes_from_mat(mat_out, 'output', d_net_nodes=d_net_)
    d_networks, _ = set_nodes_from_mat(mat_net, 'network', d_net_nodes=d_net_)

    # distribute frequency among input nodes,
    d_inputs, set_freqs = set_frequencies(d_inputs, {0}, 1)

    # distribute frequencies among network nodes
    d_networks, set_freqs = set_frequencies(d_networks, set_freqs, capacity, l_=d_inputs.values())

    # Finally, build nodes
    l_inputs = [nodes.InputNode(k_, 'input', FrequencyStack(len(set_freqs) * 2, v_['freqs']), v_['children'])
                for k_, v_ in sorted(d_inputs.items(), key=lambda (k, v): k)]
    l_networks = [
        nodes.NetworkNode(k_, 'network', FrequencyStack(len(set_freqs) * 2, v_['freqs']), v_['children'], l0, tau)
        for k_, v_ in sorted(d_networks.items(), key=lambda (k, v): k)
        ]
    l_outputs = [nodes.OutputNode(k_, 'output', v_['parents'])
                 for k_, v_ in sorted(d_outputs.items(), key=lambda (k, v): k)]

    return l_inputs, l_outputs, l_networks


def set_nodes_from_mat(mat, key, d_net_nodes={}):
    d_nodes = {}
    d_net_nodes = d_net_nodes

    if key == 'input':
        x_coord, y_coord = mat.nonzero()
        for i in set(x_coord):
            d_nodes[i] = {'children': [], 'freqs': []}
            for j in y_coord[x_coord == i]:
                d_nodes[i]['children'] += [('network_{}'.format(j), mat[i, j])]

                if j not in d_net_nodes.keys():
                    d_net_nodes[j] = {'children': [], 'freqs': []}

    elif key == 'output':
        x_coord, y_coord = mat.nonzero()
        for j in set(y_coord):
            d_nodes[j] = {'parents': [], 'freqs': []}
            for i in x_coord[y_coord == j]:
                d_nodes[j]['parents'] += [('network_{}'.format(i), mat[i, j])]

                if i not in d_net_nodes.keys():
                    d_net_nodes[i] = {'children': [], 'freqs': []}

    elif key == 'network':
        x_coord, y_coord = mat.nonzero()
        for i in set(x_coord):

            if i in d_net_nodes.keys():
                d_net_nodes.pop(i)
            d_nodes[i] = {'children': [], 'freqs': []}
            for j in y_coord[x_coord == i]:
                d_nodes[i]['children'] += [('network_{}'.format(j), mat[i, j])]

        d_nodes.update(d_net_nodes)

    else:
        raise ValueError('Choose key between \'input\', \'output\' or \'network\'')

    return d_nodes.copy(), d_net_nodes


def set_frequencies(d_nodes, freqs, capacity, l_=None):

    for k, v in d_nodes.items():

        l_children = [t[0] for t in v['children']]

        if len(l_children) == 0:
            freqs_ = list(freqs)[:capacity]

        else:
            # Look for sibling
            for child in l_children:
                _freqs = filter(lambda (k_, v_): child in [t[0] for t in v_['children']] and k != k_, d_nodes.items())
                _freqs = sum(map(lambda (k_, v_): v_['freqs'], _freqs), [])

                if l_ is not None:
                    _freqs += sum(map(lambda v_: v_['freqs'], filter(lambda v_: child in [t[0] for t in v_['children']],
                                                                     l_)), [])

                freqs_ = freqs.difference(set(_freqs))

        if len(freqs_) >= capacity:
            d_nodes[k]['freqs'] = list(freqs_)[:capacity]

        else:
            # Compute new frequencies
            new_freqs = range(max(freqs) + 1, max(freqs) + capacity + 1 - len(freqs_))

            # Update available freq
            freqs = freqs.union(set(new_freqs))

            d_nodes[k]['freqs'] = list(set(new_freqs).union(set(freqs_)))

    return d_nodes, freqs
