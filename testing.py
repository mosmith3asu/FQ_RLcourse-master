import numpy as np
from dataclasses import dataclass
from FIS import FuzzyInferenceSystem
from Environment import Environment
# def struct():
def emmpty_obj():
    class EmptyObj(object): pass
    class_instance = EmptyObj()
    return class_instance

@dataclass
class FQL_Config:
    num_episodes: int = 50
    max_epi_duration: int = 30
    fs_admittance: int = 100
    fs_learning: int = 10

    FQL: object = type('FQL_obj', (object,), {})
    FQL.v_range,    FQL.v_bins      = [-2, 4], 4
    FQL.F_range,    FQL.F_bins      = [-8, 8], 4
    FQL.dvdt_range, FQL.dvdt_bins   = [-5, 5], 4
    FQL.dFdt_range, FQL.dFdt_bins   = [-6, 6], 4


# print(porsche)
Config = FQL_Config()
print(Config.FQL.v_range)