# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 12:56:38 2023

@author: https://youtu.be/5Kr31IFwJiI?si=oyoe9OkQQLzR6Dn2
"""
from sklearn import datasets, model_selection
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, IBMQ
import numpy as np
from sklearn import svm
from qiskit import BasicAer
import copy as copy
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris.data[:100]
y = iris.target[:100]
x_train, x_test, y_train, y_test =  model_selection.train_test_split(x,y,test_size=0.33, random_state=42)

for i in range(len(x)):
    print(str(x[i]) + " -> " + str(y[i]))

N = 4

def feature_map(x):
    q = QuantumRegister(N)
    c = ClassicalRegister(1)
    qc = QuantumCircuit(q,c)

    for i, x in enumerate(x_train[0]):
        qc.rx(x,i)
    return qc,c

def variational_circuit(qc, theta):
    for i in range(N-1):
        qc.cnot(i, i+1) #creates entanglement
    qc.cnot(N-1, 0)
    for i in range(N):
        qc.ry(theta[i], i)
    return qc

def quantum_nn(x, theta, simulator = True):
    
    qc, c = feature_map(x)
   
    qc.barrier()
    qc = variational_circuit(qc, theta)
    
    qc.barrier()
    qc.measure(0,c)
    
    #return qc

#def train_nn(qc):
    shots = 1024     #1E4 = running 10K times
    backend = BasicAer.get_backend('qasm_simulator')
    
    if simulator == False:
        shots = 5000
        provider = IBMQ.load_account()
        backend = provider.get_backend('ibmq_athens')
    
    jobs = execute(qc, backend, shots = shots)
    result = jobs.result()
    counts = result.get_counts(qc)

    #qc.draw('mpl')
    
    return counts['1']/shots
    

    
def loss_fn(pred, target):
    return(pred-target)**2

def gradient(x,y,theta):
    delta = 0.01
    grad = []
    
    for i in range(len(theta)):
        dtheta = copy.copy(theta)
        dtheta[i]+= delta
        
        pred1 = quantum_nn(x, dtheta)
        pred2 = quantum_nn(x, theta)
        
        grad.append((loss_fn(pred1,y)-loss_fn(pred2, y))/ delta)
    return np.array(grad)

def accuracy(x,y, theta):
    counter = 0
    
    for x_i, y_i in zip(x,y):
        prediction =  quantum_nn(x_i, theta)
        
        if prediction <=0.5 and y_i ==0:
            counter += 1
        elif prediction >= 0.5 and y_i ==1:
            counter +=1
        
        return counter/len(y)
    

N=4
lr=0.05
loss_list=[]
epochs=12
theta=np.ones(N)

print('Epoch \t Loss \t Accuracy')

for i in range(epochs):
    loss_tmp=[]
    for X_i, Y_i in zip(x_train, y_train):
        pred=quantum_nn(X_i, theta)
        loss_tmp.append(loss_fn(pred, Y_i))
        theta=theta-lr*gradient(X_i, Y_i, theta)

    loss_list.append(np.mean(loss_tmp))
    acc=accuracy(x_train, y_train, theta)

    print(f'{i} \t {loss_list[-1]:.3f} \t {acc:.3f}')
    
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show

test_accuracy = accuracy(x_test, y_test, theta)
print("Test accuracy = " + str(test_accuracy))

# Comparison with classical NN
clf = svm.SVC()
clf.fit(x_train, y_train)

print("Comparing SVC prediction against actual data")
print(clf.predict(x_test))
print(y_test)

# Run on real hardware
quantum_nn(x_test[0], theta, simulator=False)
quantum_nn(x_test[0], theta)
