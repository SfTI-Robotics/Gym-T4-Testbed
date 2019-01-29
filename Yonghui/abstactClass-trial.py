import abc

class AbstractAnimal(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def walk(self):
        ''' data '''

    @abc.abstractmethod
    def talk(self):
        ''' data '''

    @abc.abstractmethod
    def laugh(self):
        ''' data '''

class Duck(AbstractAnimal):
    name = ''

    def __init__(self, name):
        print('duck created.')
        self.name = name

    def walk(self):
        print('walks')

    def laugh(self):
        print('laugh')
    
    def talk(self):
        print('quack')
        Duck.laugh(self)

    

obj = Duck('duck1')
obj.talk()
obj.walk()

# ==============================================================
