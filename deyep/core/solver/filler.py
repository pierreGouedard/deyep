# Global import
from scipy.sparse import csc_matrix, hstack, vstack

# Local import
from deyep.core.datastructures.deep_network_dry import DeepNetworkDry
from deyep.core.datastructures.deep_network import DeepNetwork
from deyep.core.runner.comon import DeepNetRunner
from deyep.utils.driver.nmp import NumpyDriver
from deyep.core.imputer.array import DoubleArrayImputer
from deyep.core.solver.utils import DeepNetUtils


class DeepNetFiller(object):
    def __init__(self, deep_network, imputer, params_network, driver=NumpyDriver()):

        self.params_network, self.driver, self.tmpdir = params_network, driver, None

        # Original system
        self.deep_network_original, self.imputer_original = DeepNetworkDry.from_deep_network(deep_network), imputer

        # Get mapping from network node to output in original network
        self.d_outputs = DeepNetFiller.get_output_mapping(deep_network)

        # Init
        self.d_mapping, self.deep_network_fill, self.imputer_fill = None, None, None

    @staticmethod
    def get_output_mapping(dn):
        d_outputs = {}
        for n in dn.output_nodes:
            for (name, _) in n.parents:
                id_ = int(name.split('_')[-1])
                d_outputs[id_] = d_outputs.get(id_, []) + [n.id]

        return d_outputs

    def build_imputer_fill(self, d=2):

        # Init stream imputer
        self.imputer_original.stream_features(is_cyclic=False)

        # Create Runner
        runner = DeepNetRunner(self.deep_network_original, self.params_network['depth'] + 1, self.imputer_original)

        # Get node activation as well as output GOT
        sax_sn, d_map = runner.track_activity(self.params_network['depth'], l_nodes=self.d_outputs.keys())
        sax_so = self.imputer_original.features_backward

        self.d_mapping, sax_out = {}, csc_matrix((sax_so.shape[0], 0))

        for id_, l_outputs in self.d_outputs.items():
            # Get signal of interset
            sax_sn_sub = self.explode_level(sax_sn[:, d_map[id_]])
            sax_so_sub = sax_so[:, l_outputs]

            # Get level information
            d_level = self.deep_network_original.network_nodes[id_].d_levels
            l_max = self.deep_network_original.D[:, id_].astype(int).sum()
            d_level_sub = {k: v for k, v in d_level.items() if (d_level[k] and k < l_max) or k == l_max}

            # Update output
            sax_out_sub, d_level_out = self.build_output(sax_sn_sub, sax_so_sub.astype(int), d_level_sub)

            if sax_out_sub.nnz > 0:
                sax_out = hstack([sax_out, sax_out_sub])
                self.d_mapping[id_] = [{
                    'id': sax_out.shape[1] - 1,
                    'level': DeepNetUtils.transform_level(d_level_out, self.params_network['l0'])
                }]

            sax_out_sub, d_level_out = self.build_output(sax_sn_sub, sax_so_sub.astype(int), d_level_sub, init=False)

            if sax_out_sub.nnz > 0:
                sax_out = hstack([sax_out, sax_out_sub])
                self.d_mapping[id_] = self.d_mapping.get(id_, []) + [{
                    'id': sax_out.shape[1] - 1,
                    'level': DeepNetUtils.transform_level(d_level_out, self.params_network['l0'])
                }]

        # Create imputer
        if sax_out.sum() > 0:
            self.create_imputer(sax_out.astype(bool))
        else:
            self.imputer_fill = None

        return self

    @staticmethod
    def explode_level(sax_s):
        sax_s_ = csc_matrix((sax_s.shape[0], sax_s.max()))

        for i in range(1, sax_s.max() + 1):
            sax_s_[:, i - 1] = (sax_s == i)

        return sax_s_

    @staticmethod
    def build_output(sax_s, sax_o, d_level, init=True, offset=100):
        n, sax_out, sax_, d_level_out, d_ = 0, csc_matrix((sax_o.shape[0], 1)), csc_matrix((sax_o.shape[0], 1)), {}, {}

        for i in range(min(d_level.keys()), max(d_level.keys()) + 1):

            # Get state
            state = init if n % 2 == 0 else not init

            if (d_level.get(i, False) == state) or d_level.get(i, False):

                if d_level.get(i, False) == state and not d_level.get(i, False):
                    sax_out += sax_
                    d_level_out.update(d_)

                sax_s_sub, d_[i], sax_r = sax_s[:, i - 1].transpose(), state, csc_matrix(sax_out.shape)

                for sax_o_sub in sax_o.transpose():
                    sax_r += (2 * (sax_o_sub - sax_s_sub) + sax_s_sub).multiply(sax_s_sub).transpose()

                if state:
                    sax_ += (sax_r < 0).astype(int)
                else:
                    sax_ += (sax_r > 0).astype(int)

            else:
                sax_, d_ = csc_matrix((sax_o.shape[0], 1)), {}

            n += 1

        # Fill upper level
        state, l_max = init if n % 2 == 0 else not init,  max(d_level.keys())
        d_level_out = {k: d_level_out.get(k, d_.get(k, d_level.get(k, False))) for k in range(l_max + 1)}
        for n in range(l_max + 1, offset + 1):
            d_level_out[n] = state

        return sax_out + sax_ > 0, d_level_out

    def create_imputer(self, sax_output):

        # Create tmp dir
        if self.tmpdir is not None:
            self.remove_tmpdir()
        self.tmpdir = self.driver.TempDir(suffix='_filler_out', create=True)

        # Create I/O and save it into tmpdir files
        self.imputer_fill = DoubleArrayImputer('filler', self.imputer_original.dirin, self.tmpdir.path)
        self.imputer_fill.read_raw_data('forward.npz', 'backward.npz')
        self.imputer_fill.run_preprocessing()
        self.imputer_fill.write_features('forward.npz', 'backward.npz')

        # Replace standard output with new output
        self.driver.write_file(sax_output, self.driver.join(self.tmpdir.path, 'backward.npz'), is_sparse=True)
        self.imputer_fill.read_features()

    def remove_tmpdir(self):
        self.tmpdir.remove()
        self.tmpdir = None

    def build_graph_fill(self, builder, rid):
        no = sum([len(l_outs) for l_outs in self.d_mapping.values()])

        if self.params_network['depth'] > 1:

            # Instantiate builder
            builder = builder(
                self.params_network['ni'], self.params_network['nd'], 1, self.params_network['depth'],
                self.params_network['p0'], self.params_network['w0'], self.params_network['l0'],
                self.params_network['tau'], self.params_network['capacity']
            )

            dno = builder.build_network('filler_tmp_o')
            builder.reset_structure()
            dne = builder.build_network('filler_tmp_e')
            sax_I, _, sax_D, _, _ = DeepNetUtils.merge_matrices(dno, dne)

            #  Build outputs and candidate
            sax_O, sax_Cm = csc_matrix((sax_D.shape[0], no)), csc_matrix((sax_D.shape[0], no))

            # Make sure that both graph are output disjoint
            l_io, l_ie = [x[0]['id'] for x in self.d_mapping.values()], \
                [x[1]['id'] for x in self.d_mapping.values() if len(x) > 1]

            sax_Cm[:dno.D.shape[0], sorted(l_ie)], sax_Cm[dno.D.shape[0]:, sorted(l_io)] = True, True

            self.deep_network_fill = DeepNetwork.from_matrices(
                'filler_{}'.format(self.params_network['depth']), sax_D, sax_I, sax_O, self.params_network['capacity'],
                network_id='network_{}'.format(rid), w0=self.params_network['w0'], l0=self.params_network['l0'],
                tau=self.params_network['tau'], Cm=sax_Cm
            )

            # Update mapping with size of each filling network aded
            self.d_mapping['ido'],  self.d_mapping['ide'] = [n.id for n in dno.network_nodes], \
                [n.id for n in self.deep_network_fill.network_nodes[len(dno.network_nodes):]]

        else:
            graph = {
                'Iw': csc_matrix((0, 0)), 'Dw': csc_matrix((0, 0)), 'Ow': csc_matrix((0, 0)),
                'Cm': csc_matrix((0, 0)), 'IOw': csc_matrix(self.params_network['ni'], no)
            }

            self.deep_network_fill = DeepNetwork(
                'filler_{}'.format(self.params_network['depth']), self.params_network['w0'], self.params_network['l0'],
                self.params_network['tau'], [], [], [], graph, 0, 0, network_id='network_{}'.format(rid)
            )

        return self

