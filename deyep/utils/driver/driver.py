import os


class Driver(object):

    def __init__(self, name, desc):
        self.name = name
        self.desc = desc

    def __str__(self):
        return '{} - {}'.format(self.name, sefl.desc)

    def read_file(self, url, **kwargs):
        raise NotImplementedError

    def write_file(self, url, **kwargs):
        raise NotImplementedError

    def join(self, arg, *args):
        raise NotImplementedError


class FileDriver(Driver):

    def __init__(self, name, desc):
        Driver.__init__(self, name, desc)

    def join(self, arg, *args):
        return os.path.join(arg, *args)


class HdhfsDriver(Driver):

    def __init__(self, name, desc):
        Driver.__init__(self, name, desc)


class SqlDriver(Driver):

    def __init__(self, name, desc):
        Driver.__init__(self, name, desc)
