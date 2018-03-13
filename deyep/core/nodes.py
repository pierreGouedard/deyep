

class Node(object):

    def __init__(self, id, value, statue):
        self.id = id
        self.value = value
        self.statue = statue

    def to_dict(self):
        return {'id': self.id, 'value': self.value, 'statue': self.statue}


class InputNode(Node):

    def __init__(self, id, value, statue, frequency_range, children):
        Node.__init__(self, 'input_{}'.format(id), value, statue)
        self.frequency_range = frequency_range
        self.frequency_stack = None
        self.children = children

    def process_forward(self):
        raise NotImplementedError

    def process_backward(self):
        raise NotImplementedError

    def to_dict(self):
        d_out = Node.to_dict(self)
        d_out.update({'frequency_range': self.frequency_range, 'frequency_stack': self.frequency_stack,
                      'children': self.children})
        return d_out


class OutputNode(Node):

    def __init__(self, id, value, statue):
        Node.__init__(self, 'output_{}'.format(id), value, statue)

    def process_forward(self):
        raise NotImplementedError

    def process_backward(self):
        raise NotImplementedError

    def to_dict(self):
        return Node.to_dict(self)


class NetworkNode(Node):

    def __init__(self, id, value, statue, frequency_range, children):
        Node.__init__(self, 'network_{}'.format(id),  value, statue)

        self.frequency_range = frequency_range
        self.level = -1
        self.frequency_map = dict()
        self.frequency_stack = None
        self.children = children

    def process_forward(self):
        raise NotImplementedError

    def process_backward(self):
        raise NotImplementedError

    def update_level(self):
        raise NotImplementedError

    def to_dict(self):
        d_out = Node.to_dict(self)
        d_out.update({'frequency_range': self.frequency_range, 'level': self.level,
                      'frequency_map': self.frequency_map, 'frequency_stack': self.frequency_stack,
                      'children': self.children})
        return d_out
