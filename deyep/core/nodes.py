

class Node(object):

    def __init__(self, id, type, values):
        self.id = id
        self.values = values
        self.type = type

    def to_dict(self):
        return {'id': self.id, 'value': self.values, 'statue': self.type}


class InputNode(Node):

    def __init__(self, id, type, frequency_stack, children, values={'n_p': 0, 'n_r': 0, 'n_f': 0}):

        # Set inherited attributes
        Node.__init__(self, id, type, values)

        # Set specific attributes
        self.frequency_stack = frequency_stack
        self.children = children

    def process_forward(self):
        raise NotImplementedError

    def process_backward(self):
        raise NotImplementedError

    def to_dict(self):
        d_out = Node.to_dict(self)
        d_out.update({'frequency_stack': self.frequency_stack.to_dict(), 'children': self.children})
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

    def __init__(self, id, type, frequency_stack, children, values={'n_p': 0, 'n_r': 0, 'n_f': 0}):
        # Set inherited attributes
        Node.__init__(self, id, type, values)

        # Set specific attributes
        self.active = False
        self.level = 0
        self.frequency_stack = frequency_stack
        self.children = children

    def process_forward(self):
        raise NotImplementedError

    def process_backward(self):
        raise NotImplementedError

    def update_level(self):
        raise NotImplementedError

    def to_dict(self):
        d_out = Node.to_dict(self)
        d_out.update({'frequency_stack': self.frequency_stack.to_dict(), 'children': self.children,
                      'level': self.level})
        return d_out
