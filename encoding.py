from qiskit import QuantumCircuit
import numpy as np
from qiskit.circuit.library import ZZFeatureMap

def angle_encoding(data):
    circuit=[]

    for sample in data:
        qc= QuantumCircuit(len(sample))

        for i,value in enumerate(sample):
            qc.ry(value,i)

        circuit.append(qc)

    return circuit


def zz_encoding(data):
    circuits = []

    num_features = data.shape[1]
    feature_map = ZZFeatureMap(
        feature_dimension=num_features,
        reps=2,
        entanglement="linear"
    )

    for sample in data:
        qc = feature_map.assign_parameters(sample)
        circuits.append(qc)

    return circuits

