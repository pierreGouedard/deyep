

class Vertex(object):
    """
    Class that implement vertex of the firing graph
    """
    def __init__(self, id, type):
        self.id = id
        self.type = type

    def to_dict(self):
        return {'id': self.id, 'type': self.type}


class InputVertex(Vertex):
    """
    Class that implement input vspecific vertex
    """
    def __init__(self, id, type, basis, children):

        # Set inherited attributes
        Vertex.__init__(self, id, type)

        # Set specific attributes
        self.basis = basis
        self.children = children

    def process_forward(self):
        raise NotImplementedError

    def process_backward(self):
        raise NotImplementedError

    def to_dict(self):
        d_out = Vertex.to_dict(self)
        d_out.update({'basis': self.basis.to_dict(), 'children': self.children})
        return d_out

    @staticmethod
    def from_dict(d_node, basis):
        return InputVertex(d_node['id'], d_node['type'], basis.from_dict(d_node['basis']), d_node['children'])


class OutputVertex(Vertex):
    """
    Class that implement output vspecific vertex
    """
    def __init__(self, id, type, parents):

        # Set inherited attributes
        Vertex.__init__(self, id, type)

        # Set specific attributes
        self.parents = parents

    def process_forward(self):
        raise NotImplementedError

    def process_backward(self):
        raise NotImplementedError

    def to_dict(self):
        d_out = Vertex.to_dict(self)
        d_out.update({'parents': self.parents})
        return d_out

    @staticmethod
    def from_dict(d_node):
        return OutputVertex(d_node['id'], d_node['type'], d_node['parents'])


class CoreVertex(Vertex):
    """
    Class that implement core specific vertex
    """
    def __init__(self, id, type, basis, children, l0):
        # Set inherited attributes
        Vertex.__init__(self, id, type)

        # Set specific attributes
        self.active = False
        self.l0 = l0
        self.basis = basis
        self.children = children

    def process_forward(self):
        raise NotImplementedError

    def process_backward(self):
        raise NotImplementedError

    def update_level(self):
        raise NotImplementedError

    def to_dict(self):
        d_out = Vertex.to_dict(self)
        d_out.update({'basis': self.basis.to_dict(), 'children': self.children})
        return d_out

    @staticmethod
    def from_dict(d_node, basis):
        return CoreVertex(
            d_node['id'], d_node['type'], basis.from_dict(d_node['basis']), d_node['children'], 0
        )
