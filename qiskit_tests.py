#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:45:39 2023

@author: ernesto.acosta
"""
'''
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit import ParameterVector
from qiskit import BasicAer 
from qiskit_symb import Operator as SymbOperator


# Ansatz Class
class OperatorGenerator():
  circuit = QuantumCircuit()
  ordered_parameters = []
  
  def __init__(self, num_qubits, param_prefix):
    self.a_qreg = QuantumRegister(num_qubits)
    self.circuit = QuantumCircuit(self.a_qreg, name="mycirc")

    self.ordered_parameters = ParameterVector(param_prefix, num_qubits)
    self.circuit.rx(self.ordered_parameters[0], 0)
    self.circuit.crx(self.ordered_parameters[1], 0, 1)

  def ordered_parameters(self):
    return self.ordered_parameters

  def circuit(self):
    return self.circuit

  def get_operator(self):
    opqrnn = SymbOperator(self.circuit)
    oper = opqrnn.to_sympy()
    operArr = np.array(oper)
    
    return operArr
'''
'''
# Operador simple
og = OperatorGenerator(2,'t')
print(og.circuit)

oper = og.get_operator()
print(str(oper))

import numpy as np
v = np.array([[10],
              [20],
              [30],
              [40]])
print(str(v))

res = np.dot(oper, v)

print(str(res))

'''
'''
# Producto Matriz por Vector columna
import numpy as np

# Definir una matriz A (por ejemplo, una matriz 3x3)
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Definir un vector columna v (por ejemplo, un vector de 3 elementos)
v = np.array([[10],
              [11],
              [12]])

# Calcular el producto matricial A * v utilizando np.dot()
resultado = np.dot(A, v)

# También puedes usar el operador @ para el producto matricial
# resultado = A @ v

print(resultado)
'''
'''
# Plot función exponencial
import numpy as np
import matplotlib.pyplot as plt

# Crear un arreglo de valores x desde -π hasta π
x = np.linspace(-np.pi, np.pi, 1000)  # 1000 puntos en el intervalo

# Calcular los valores de la función seno para cada punto x
y = np.exp(x)

# Crear el gráfico de la función seno
plt.plot(x, y, label='exp(x)')

# Calcular las posiciones de las líneas verticales
vertical_lines = np.linspace(0, np.pi, 9)  # 8 líneas equidistantes entre 0 y π

# Dibujar las líneas verticales en rojo
for v_line in vertical_lines:
    plt.axvline(v_line, color='red', linestyle='--')

# Etiquetas y título del gráfico
#plt.title('Gráfico de la función seno con líneas verticales')
#plt.xlabel('x (radianes)')
#plt.ylabel('sen(x)')

# Mostrar el gráfico
plt.legend()  # Mostrar la leyenda
plt.grid(True)
plt.show()
'''
'''
#Prueba función exponencial
v = 0.9817477042468103
print(str(v))
e = np.exp(v)
print(str(e))
'''
# Execute on real computer
#!pip install ibm_quantum_experience
from qiskit import QuantumCircuit, transpile
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy

# Replace with your own IBM Quantum credentials

QISKIT_TOKEN = 'xxx' #'YOUR_API_TOKEN'
QISKIT_HUB = 'xxx'
QISKIT_GROUP = 'xxx'
QISKIT_PROJECT = 'xxx'
QISKIT_BACKEND = 'xxx'

# Initialize the account first.
IBMQ.save_account(QISKIT_TOKEN, overwrite=True)
provider = IBMQ.load_account()

available_cloud_backends = provider.backends(min_num_qubits=4, simulator=False, operational=True)
print('\n Cloud backends:')
for i in available_cloud_backends: print(i)
backend = least_busy(available_cloud_backends)
print("Least busy: " + str(backend))

# Create a Quantum Circuit acting on a quantum register of three qubits
circ = QuantumCircuit(3)
circ.h(0)
circ.cx(0,1)
circ.cx(0,2)

# Create a Quantum Circuit
meas = QuantumCircuit(3, 3)
meas.barrier(range(3))
meas.measure(range(3), range(3))

circ.add_register(meas.cregs[0])
qc = circ.compose(meas)

#drawing the circuit
qc.draw()
print("Circuit: ")
print(qc)

provider = IBMQ.get_provider(hub=QISKIT_HUB, group=QISKIT_GROUP, project=QISKIT_PROJECT)
backend = provider.get_backend(QISKIT_BACKEND)
transpiled = transpile(qc, backend=backend)
job = backend.run(transpiled)

result_sim = job.result()
counts = result_sim.get_counts(qc)
print("Result: " + str(counts))