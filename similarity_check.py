#from PyTsetlinMachineCUDA.tm import TsetlinMachine, MultiClassTsetlinMachine
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
import numpy as np 
import sympy
from sympy import sympify
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
from sympy.logic.boolalg import to_dnf, to_cnf
from sympy.logic.inference import satisfiable
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import os
import timeit
import cv2
import matplotlib.pyplot as plt
from pysat.formula import CNF
from pysat.solvers import Minisat22
import pickle 
import sys
import time
sys.path.insert(1, '../')
from robustness_utils_new import *
import random

f_tm = open("../TM_MNIST_2.pickle", "rb")
tm = pickle.load(f_tm)
f_tm.close() 

f_tm2 = open("TM_MNIST_3.pickle", "rb")
tm2 = pickle.load(f_tm2)
f_tm.close() 

f = open("../MNIST_test_1.pickle", "rb")
X_test = pickle.load(f)
f.close() 

X_test= np.array(random.choices(X_test, k=100))

number_of_classes = tm.number_of_classes
print(number_of_classes)
n_clauses = tm.number_of_clauses
number_of_features = int(tm.number_of_features/2)
weights = tm.clause_weights.reshape(tm.number_of_classes, tm.number_of_clauses)

number_of_classes2 = tm2.number_of_classes
print(number_of_classes2)
n_clauses2 = tm2.number_of_clauses
number_of_features2 = int(tm2.number_of_features/2)
weights2 = tm2.clause_weights.reshape(tm2.number_of_classes, tm2.number_of_clauses)

print(number_of_features)
print(number_of_features2)

counter = VarCounter()
x = [None] + define_variables(number_of_features, counter)
o1 = define_variables(1, counter)
o2 = define_variables(1, counter)

posnegs = []
ts_encodings = []
for t in range(number_of_classes):
    pos, neg = get_clauses(tm,t, x,weights)
    posnegs.append((pos,neg))
    n_clauses = len(pos) + len(neg) -2
    print("n_clause", n_clauses)
    encoded = encode_test(neg, pos, n_clauses,counter, o1)
    ts_encodings.append(encoded)
    
posnegs2 = []
ts_encodings2 = []
for t in range(number_of_classes):
    pos, neg = get_clauses(tm2,t, x,weights2)
    posnegs2.append((pos,neg))
    n_clauses = len(pos) + len(neg) -2
    print("n_clause", n_clauses2)
    encoded = encode_test(neg, pos, n_clauses2,counter, o2)
    ts_encodings2.append(encoded)

labels = np.zeros((number_of_classes, len(X_test)))
scores= tm.score(X_test)
for t in range(number_of_classes):
    for j in range(len(scores[t])):
        labels[t][j] = 1 if scores[t][j]>=0 else 0

print("Encoding Finished")
ps = [1,3,5]
count_number=0
statistics_robustness = np.zeros((len(ps), len(X_test),4))
from time import time
for pix,p in enumerate(ps):
    print("ps", pix, p)
    for i in range(len(X_test)):
        print("i", i)
        m_runtimes =0
        m_runtimef =0
        alert = False
        for t in range(number_of_classes):
            print("class", t)
            pos,neg = posnegs[t]
            start = time()
            runtimes, runtimef,rob = check_similarity(p, ts_encodings[t], ts_encodings2[t], x, X_test[i], labels[t][i], counter, o1, o2)
            stop = time()
            print ("train_time", stop-start)
            if runtimes >= 300:
                count_number+=1
            m_runtimes+=runtimes
            m_runtimef+=runtimef
            print("rob", rob)
            if not rob:
                alert = True
                print("break")
                break
        robust = True
        if alert:
            robust = False
        statistics_robustness[pix][i][0] =  m_runtimes/number_of_classes
        statistics_robustness[pix][i][1] = m_runtimef/number_of_classes
        statistics_robustness[pix][i][2] =  robust
        statistics_robustness[pix][i][3] = count_number
print("saving")        
f = open("MNIST_similarity_results_2_3.pickle", "wb+")
pickle.dump(statistics_robustness, f)
f.close()