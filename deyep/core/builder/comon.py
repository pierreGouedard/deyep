# Global imports
from scipy.sparse import csc_matrix
import numpy as np

# local import
import settings
from deyep.core import nodes
from deyep.core.tools.frequencies import FrequencyStack


def mat_from_tuples(l_edges, n_i, n_rn, n_o, weights='random'):

    # Init matrices
    sax_in = csc_matrix(np.zeros([n_i, n_rn]))
    sax_net = csc_matrix(np.zeros([n_rn, n_rn]))
    sax_out = csc_matrix(np.zeros([n_rn, n_o]))

    i = 0
    for (_n, n_) in l_edges:
        if 'input' in _n:
            if weights == 'random':
                v = np.random.randint(1, 100)
            else:
                v = weights[i]

            sax_in[int(_n.split('_')[1]), int(n_.split('_')[1])] = v

        elif 'network' in _n:
            if 'network' in n_:
                if weights == 'random':
                    v = np.random.randint(1, 100)
                else:
                    v = weights[i]

                sax_net[int(_n.split('_')[1]), int(n_.split('_')[1])] = v

            elif 'output' in n_:
                if weights == 'random':
                    v = np.random.randint(1, 100)
                else:
                    v = weights[i]

                sax_out[int(_n.split('_')[1]), int(n_.split('_')[1])] = v

        i += 1

    return sax_in, sax_net, sax_out


def mat_from_nodes(l_nodes):
    raise NotImplementedError


def nodes_from_mat(sax_net, sax_in, sax_out, capacity, l0=10):

    # Get dict of nodes
    d_inputs = set_nodes_from_mat(sax_in, 'input')
    d_outputs = set_nodes_from_mat(sax_out, 'output')
    d_networks = set_nodes_from_mat(sax_net, 'network')

    # distribute frequency among input nodes,
    d_inputs, set_freqs = set_frequencies(d_inputs, {0}, 1)

    # distribute frequencies among network nodes
    d_networks, set_freqs = set_frequencies(d_networks, set_freqs, capacity, l_=d_inputs.values())
    n_freq = 2 * (max(set_freqs) + 1)

    # Finally, build nodes
    l_inputs = [nodes.InputNode(k_, 'input', FrequencyStack(len(set_freqs) * 2, v_['freqs']), v_['children'])
                for k_, v_ in sorted(d_inputs.items(), key=lambda (k, v): k)]
    l_networks = [
        nodes.NetworkNode(k_, 'network', FrequencyStack(len(set_freqs) * 2, v_['freqs']), v_['children'], l0)
        for k_, v_ in sorted(d_networks.items(), key=lambda (k, v): k)
        ]
    l_outputs = [nodes.OutputNode(k_, 'output', v_['parents'])
                 for k_, v_ in sorted(d_outputs.items(), key=lambda (k, v): k)]

    return l_inputs, l_outputs, l_networks, n_freq


def set_nodes_from_mat(mat, key):

    if key == 'input':
        d_nodes = {i: {'children': [], 'freqs': []} for i in range(mat.shape[0])}
        for i, j in zip(*mat.nonzero()):
            d_nodes[i]['children'] += [('network_{}'.format(j), mat[i, j])]

    elif key == 'output':
        d_nodes = {i: {'parents': [], 'freqs': []} for i in range(mat.shape[1])}
        for i, j in zip(*mat.nonzero()):
            d_nodes[j]['parents'] += [('network_{}'.format(i), mat[i, j])]

    elif key == 'network':
        d_nodes = {i: {'children': [], 'freqs': []} for i in range(mat.shape[0])}
        for i, j in zip(*mat.nonzero()):
            d_nodes[i]['children'] += [('network_{}'.format(j), mat[i, j])]
    else:
        raise ValueError('Choose key between \'input\', \'output\' or \'network\'')

    return d_nodes.copy()


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


def gather_matrices(ax_in, ax_net, ax_out):

    ax_dn = np.vstack((ax_in, ax_net))

    ax_dn = np.hstack((np.zeros((ax_dn.shape[0], ax_in.shape[0])), ax_dn))

    ax_out_ = np.vstack((np.zeros((ax_in.shape[0], ax_out.shape[1])), ax_out))

    ax_dn = np.hstack((ax_dn,  ax_out_))

    ax_dn = np.vstack((ax_dn, np.zeros((ax_out.shape[1], ax_dn.shape[1]))))

    return ax_dn