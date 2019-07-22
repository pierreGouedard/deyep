

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
    def __init__(self, id, type, basis, l_children):

        # Set inherited attributes
        Vertex.__init__(self, id, type)

        # Set specific attributes
        self.basis = basis
        self.children = l_children

    def to_dict(self):
        d_out = Vertex.to_dict(self)
        d_out.update({'basis': self.basis.to_dict(), 'children': self.children})
        return d_out

    @staticmethod
    def from_dict(d_vertex, basis):
        return InputVertex(d_vertex['id'], d_vertex['type'], basis.from_dict(d_vertex['basis']), d_vertex['children'])


class OutputVertex(Vertex):
    """
    Class that implement output vspecific vertex
    """
    def __init__(self, id, type, d_parents):

        # Set inherited attributes
        Vertex.__init__(self, id, type)

        # Set specific attributes
        self.parents = d_parents

    def process_forward(self):
        raise NotImplementedError

    def process_backward(self):
        raise NotImplementedError

    def to_dict(self):
        d_out = Vertex.to_dict(self)
        d_out.update({'parents': self.parents})
        return d_out

    @staticmethod
    def from_dict(d_vertex):
        return OutputVertex(d_vertex['id'], d_vertex['type'], d_vertex['parents'])


class CoreVertex(Vertex):
    """
    Class that implement core specific vertex
    """
    def __init__(self, id, type, basis, l_children, l0):
        # Set inherited attributes
        Vertex.__init__(self, id, type)

        # Set specific attributes
        self.active = False
        self.l0 = l0
        self.basis = basis
        self.children = l_children

    def to_dict(self, **kwargs):
        d_out = Vertex.to_dict(self)
        d_out.update({'basis': self.basis.to_dict(), 'children': self.children})
        d_out.update(kwargs)
        return d_out

    @staticmethod
    def from_dict(d_vertex, basis):
        return CoreVertex(
            d_vertex['id'], d_vertex['type'], basis.from_dict(d_vertex['basis']), d_vertex['children'],
            d_vertex.get('l0', 0)
        )
