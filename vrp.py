import time
from typing import List
from dataclasses import dataclass


@dataclass
class Customer:
    number: int
    lat: int
    lng: int
    q: int              # demand
    ei: int             # ready time
    li: int             # due date
    service_time: int

    def __key__(self):
        return (self.x, self.y)

    def __hash__(self):             # enables
        return hash(self.__key__())

    def __eq__(self, other):
        if isinstance(other, Customer):
            return self.__key() == other.__key()
        return NotImplemented

    def __repr__(self) -> str:
        return f'{self.name}'

    def __str__(self) -> str:
        return self.__repr__()


@dataclass
class Algo:
    C: List[Customer]
    D: List[list]
    T: int
    Q: int
    start = time.time()
    t = 0
