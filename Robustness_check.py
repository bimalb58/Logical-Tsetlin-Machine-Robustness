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
import os
import timeit
import cv2
import pickle 
import sys
import time
sys.path.insert(1, '../')
from robustness_utils_new import *
import random

f_tm = open("TM_MNIST.pickle", "rb")
tm = pickle.load(f_tm)
f_tm.close() 

f = open("MNIST_test.pickle", "rb")
X_test = pickle.load(f)
f.close() 

X_test= np.array(random.choices(X_test, k=200))

number_of_classes = tm.number_of_classes
print(number_of_classes)
n_clauses = tm.number_of_clauses
number_of_features = int(tm.number_of_features/2)
weights = tm.clause_weights.reshape(tm.number_of_classes, tm.number_of_clauses)

counter = VarCounter()
x = [None] + define_variables(number_of_features, counter)
o = define_variables(1, counter)

posnegs = []
ts_encodings = []
for t in range(number_of_classes):
    pos, neg = get_clauses(tm,t, x,weights)
    posnegs.append((pos,neg))
    n_clauses = len(pos) + len(neg) -2
    print("n_clause", n_clauses)
    encoded = encode_test(neg, pos, n_clauses,counter, o)
    ts_encodings.append(encoded)

labels = np.zeros((number_of_classes, len(X_test)))
scores= tm.score(X_test)
for t in range(number_of_classes):
    for j in range(len(scores[t])):
        labels[t][j] = 1 if scores[t][j]>=0 else 0
        
print("Encoding Finished")
ps = [1, 3, 5]

statistics_robustness = np.zeros((len(ps), len(X_test),3))
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
            runtimes, runtimef,rob = check_robustness(p, x, ts_encodings[t], X_test[i], labels[t][i],counter, o)
            stop = time()
            print ("rob_solve_time", stop-start)
            m_runtimes+=runtimes
            m_runtimef+=runtimef
            print("rob", rob)
            robust = rob
            if rob == False:
                print("break")
                break
        statistics_robustness[pix][i][0] =  m_runtimes/number_of_classes
        statistics_robustness[pix][i][1] = m_runtimef/number_of_classes
        statistics_robustness[pix][i][2] =  robust
print("saving")        
f = open("MNIST_robustness_result.pickle", "wb+")
pickle.dump(statistics_robustness, f)
f.close()