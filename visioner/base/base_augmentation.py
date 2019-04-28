
from abc import ABC, abstractmethod

class BaseAugmentation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass

    