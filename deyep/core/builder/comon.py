# Global imports
from scipy.sparse import lil_matrix, csc_matrix
import numpy as np

# local import
from deyep.core import nodes
from deyep.core.tools.basis.fourrier import FourrierBasis
from deyep.core.tools.basis.canonical import CanonicalBasis


def mat_from_tuples(l_edges, n_i, n_rn, n_o, weights='random'):

    # Init matrices
    sax_in = lil_matrix(np.zeros([n_i, n_rn]))
    sax_net = lil_matrix(np.zeros([n_rn, n_rn]))
    sax_out = lil_matrix(np.zeros([n_rn, n_o]))

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

    return sax_in.tocsc(), sax_net.tocsc(), sax_out.tocsc()


def mat_from_nodes(l_nodes):
    raise NotImplementedError


def nodes_from_mat(sax_net, sax_in, sax_out, capacity, basis, l0=10, fixed_weight=None):

    # Get dict of nodes
    d_inputs = set_nodes_from_mat(sax_in, 'input')
    d_outputs = set_nodes_from_mat(sax_out, 'output')
    d_networks = set_nodes_from_mat(sax_net, 'network')

    # distribute frequency among input nodes,
    d_inputs, set_freqs, d_forward_freqs = set_frequencies(d_inputs, {0}, 1, {})

    # distribute frequencies among network nodes
    d_networks, set_freqs_, d_forward_freqs = set_frequencies(d_networks, {0}, capacity, d_forward_freqs,
                                                              offset=len(set_freqs))

    n_freq = ((max(set_freqs_) + 1) * capacity) + len(set_freqs)

    # Finally, build nodes
    if basis == 'fourrier':
        l_inputs = [
            nodes.InputNode(
                k_, 'input', FourrierBasis(min(v_['freqs']), n_freq, {}, 1),
                zip(v_['children_name'], v_['children_weight'])
            ) for k_, v_ in sorted(d_inputs.items(), key=lambda (k, v): k)
            ]
        l_networks = [
            nodes.NetworkNode(
                k_, 'network',
                FourrierBasis(min(v_['freqs']), n_freq, d_forward_freqs.get('network_{}'.format(k_), []), capacity),
                zip(v_['children_name'], v_['children_weight']), l0
            ) for k_, v_ in sorted(d_networks.items(), key=lambda (k, v): k)
            ]
    elif basis == 'canonical':
        l_inputs = [
            nodes.InputNode(k_, 'input', CanonicalBasis(min(v_['freqs']), n_freq, {}, 1),
                            zip(v_['children_name'], v_['children_weight'])) for k_, v_ in sorted(d_inputs.items(), key=lambda (k, v): k)
            ]
        l_networks = [
            nodes.NetworkNode(
                k_, 'network',
                CanonicalBasis(min(v_['freqs']), n_freq, d_forward_freqs.get('network_{}'.format(k_), []), capacity),
                zip(v_['children_name'], v_['children_weight']), l0
            ) for k_, v_ in sorted(d_networks.items(), key=lambda (k, v): k)
            ]
    else:
        raise ValueError('basis not understood: {}, choose between "fourrier" and "canonical"')
    l_outputs = [nodes.OutputNode(k_, 'output', zip(v_['parents_name'], v_['parents_weight']))
                 for k_, v_ in sorted(d_outputs.items(), key=lambda (k, v): k)]

    return l_inputs, l_outputs, l_networks, n_freq


def set_nodes_from_mat(mat, key):

    if key == 'input':
        d_nodes = {i: {'children_name': [], 'children_weight': [], 'freqs': []} for i in range(mat.shape[0])}
        for i, j in zip(*mat.nonzero()):
            d_nodes[i]['children_name'] += ['network_{}'.format(j)]
            d_nodes[i]['children_weight'] += [mat[i, j]]

    elif key == 'output':
        d_nodes = {i: {'parents_name': [], 'parents_weight': [], 'freqs': []} for i in range(mat.shape[1])}
        for i, j in zip(*mat.nonzero()):
            d_nodes[j]['parents_name'] += ['network_{}'.format(i)]
            d_nodes[j]['parents_weight'] += [mat[i, j]]

    elif key == 'network':
        d_nodes = {i: {'children_name': [], 'children_weight': [], 'freqs': []} for i in range(mat.shape[0])}
        for i, j in zip(*mat.nonzero()):
            d_nodes[i]['children_name'] += ['network_{}'.format(j)]
            d_nodes[i]['children_weight'] += [mat[i, j]]
    else:
        raise ValueError('Choose key between \'input\', \'output\' or \'network\'')

    return d_nodes.copy()


def set_frequencies(d_nodes, freqs, capacity, d_forward_freqs, offset=0):

    for k, v in d_nodes.items():
        print k
        if len(v['children_name']) > 0:
            # Look for sibling
            freqs_ = freqs.difference(get_occupied_freqs(k, v['children_name'], d_nodes))

            if len(freqs_) > 0:
                d_nodes[k]['freqs'] = [list(freqs_)[0]]

            else:
                # Add new frequencies
                d_nodes[k]['freqs'] = [max(freqs) + 1]
                freqs = freqs.union([max(freqs) + 1])
        else:
            d_nodes[k]['freqs'] = [list(freqs)[0]]

        if d_forward_freqs is not None:
            for child in v['children_name']:
                d_forward_freqs[child] = d_forward_freqs.get(child, []) + [offset + (d_nodes[k]['freqs'][0] * capacity)]

    for k, v in d_nodes.items():
        v['freqs'] = [offset + (min(v['freqs']) * capacity)]

    return d_nodes, freqs, d_forward_freqs


def get_occupied_freqs(k, l_children, d_nodes):
    occupied_freqs = []
    for child in l_children:
        for k_, v_ in d_nodes.items():
            if child in v_['children_name'] and k != k_ and len(v_['freqs']) > 0:
                occupied_freqs += v_['freqs']

    return set(occupied_freqs)


def gather_matrices(ax_in, ax_net, ax_out):

    ax_dn = np.vstack((ax_in, ax_net))

    ax_dn = np.hstack((np.zeros((ax_dn.shape[0], ax_in.shape[0])), ax_dn))

    ax_out_ = np.vstack((np.zeros((ax_in.shape[0], ax_out.shape[1])), ax_out))

    ax_dn = np.hstack((ax_dn,  ax_out_))

    ax_dn = np.vstack((ax_dn, np.zeros((ax_out.shape[1], ax_dn.shape[1]))))

    return ax_dn