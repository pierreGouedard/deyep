

class Node(object):

    def __init__(self, id, type, values):
        self.id = id
        self.values = values
        self.type = type

    def to_dict(self):
        return {'id': self.id, 'value': self.values, 'statue': self.type}


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


class NetworkNode(Node):

    def __init__(self, id, type, basis, children, l0,  values={'n_p': 0, 'n_r': 0, 'n_f': 0}):
        # Set inherited attributes
        Node.__init__(self, id, type, values)

        # Set specific attributes
        self.active = False
        self.d_levels = {i: l0 for i in range(1, 100)}
        self.d_levels.update({0: 0})
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
