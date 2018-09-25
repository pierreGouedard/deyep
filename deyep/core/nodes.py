

class Node(object):

    def __init__(self, id, type, values):
        self.id = id
        self.values = values
        self.type = type

    def to_dict(self):
        return {'id': self.id, 'values': self.values, 'type': self.type}


class InputNode(Node):

    def __init__(self, id, type, basis, children, values={'n_p': 0, 'n_r': 0, 'n_f': 0}):

        # Set inherited attributes
        Node.__init__(self, id, type, values)

        # Set specific attributes
        self.basis = basis
        self.children = children

    def process_forward(self):
        raise NotImplementedError

    def process_backward(self):
        raise NotImplementedError

    def to_dict(self):
        d_out = Node.to_dict(self)
        d_out.update({'basis': self.basis.to_dict(), 'children': self.children})
        return d_out

    @staticmethod
    def from_dict(d_node, basis):
        return InputNode(d_node['id'], d_node['type'], basis.from_dict(d_node['basis']), d_node['children'])


class OutputNode(Node):

    def __init__(self, id, type, parents):

        # Set inherited attributes
        Node.__init__(self, id, type, None)

        # Set specific attributes
        self.parents = parents

    def process_forward(self):
        raise NotImplementedError

    def process_backward(self):
        raise NotImplementedError

    def to_dict(self):
        d_out = Node.to_dict(self)
        d_out.update({'parents': self.parents})
        return d_out

    @staticmethod
    def from_dict(d_node):
        return OutputNode(d_node['id'], d_node['type'], d_node['parents'])


class NetworkNode(Node):

    def __init__(self, id, type, basis, children, l0,  values={'n_p': 0, 'n_r': 0, 'n_f': 0}, level=None):
        # Set inherited attributes
        Node.__init__(self, id, type, values)

        # Set specific attributes
        self.active = False

        if level is None:
            self.d_levels = {i: l0 for i in range(1, 100)}
            self.d_levels.update({0: 0})
        else:
            self.d_levels = level

        self.basis = basis
        self.children = children

    def process_forward(self):
        raise NotImplementedError

    def process_backward(self):
        raise NotImplementedError

    def update_level(self):
        raise NotImplementedError

    def to_dict(self):
        d_out = Node.to_dict(self)
        d_out.update({'basis': self.basis.to_dict(), 'children': self.children,
                      'level': self.d_levels})
        return d_out

    @staticmethod
    def from_dict(d_node, basis):
        return NetworkNode(d_node['id'], d_node['type'], basis.from_dict(d_node['basis']), d_node['children'], 0,
                           level=d_node['level'])

