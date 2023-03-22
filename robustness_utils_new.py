import numpy as np
import os
import timeit
import cv2
from pysat.solvers import Minisat22
from pysat.solvers import Glucose4
from pysat.formula import CNF
from threading import Timer

class VarCounter:
    def __init__(self):
        self.c = 0
    # getter method
    def get(self):
        return self.c
      
    # setter method
    def set(self, x):
        self.c = x


def interrupt(s):
    s.interrupt()

 
def is_satisfiable(cnf):
    m = Glucose4(bootstrap_with=cnf)
    
    timer = Timer(600, interrupt, [m])
    timer.start()

    #s = m.solve_limited()
    s = m.solve_limited(expect_interrupt=True)

    m.clear_interrupt()
    m.delete()
    
    return s

def define_variables(number,counter):
    variable_counter = counter.get()
    #it outputs a vector of *number* variables
    V = [i for i in range(variable_counter+1,variable_counter+number+1)]
    counter.set(variable_counter+number)
    return V

def get_clauses(tm,tm_class,x,weights):
    pos_clause = [None]
    neg_clause = [None]
    n_clauses = tm.number_of_clauses
    number_of_features = int(tm.number_of_features/2)
    
    #Positive Clauses
    for j in range(0, n_clauses, 2):
        variables=[]
        for k in range(number_of_features*2):
            if tm.ta_action(tm_class, j, k) == 1:
                if k < number_of_features:
                    variables.append(x[k+1])
                else:
                    variables.append(-x[k+1-number_of_features])
        #print(weights[tm_class,j])
        #for _ in range(weights[tm_class,j]):
        pos_clause.append(variables)

    #Negative Clauses
    for j in range(1, n_clauses, 2):
        variables=[]
        for k in range(number_of_features*2):
            if tm.ta_action(tm_class, j, k) == 1:
                if k < number_of_features:
                    variables.append(x[k+1])
                else:
                    variables.append(-x[k+1-number_of_features])
        #for _ in range(weights[tm_class,j]):
        neg_clause.append(variables)
        
    return pos_clause, neg_clause

def seq_counter(l,r,K):
    # l has the form [0,l1,l2,l3....]
    L = len(l)-1
    
    #first conjunct 
    forlist = [[-l[1], r[1][1]], [-r[1][1], l[1]]]
    #second conjunct
    for  j in range(2,K+1):
        forlist.append( [-r[1][j]] )
    #third conjunct
    for i in range(2,L+1):
        forlist.append( [r[i][1], -l[i]] )
        forlist.append( [r[i][1], -r[i-1][1]] )
        forlist.append( [-r[i][1], l[i], r[i-1][1]] )
    #fourth conjunct
        for j in  range(2,K+1):
            forlist.append( [-r[i-1][j-1], r[i][j], -l[i]] )
            forlist.append( [l[i], r[i-1][j], -r[i][j]] )
            forlist.append( [r[i-1][j-1], r[i-1][j], -r[i][j]] )
            forlist.append( [-r[i-1][j], r[i][j]] )
            
    return forlist

def encode_test(pos_clause, neg_clause, n_clauses,counter, o_final):
    nclausesDiv2 = int(n_clauses/2)
    
    o = [None] + define_variables(nclausesDiv2, counter)
    
    v = [[None],[None]]
    for i in range(2):
        v[i] = v[i] + define_variables(nclausesDiv2,counter)
        
    r=[[0],[0]]
    for s in range(2):
        for i in range(1,nclausesDiv2+1):
            r[s].append([0])
            r[s][i] = r[s][i] + (define_variables(nclausesDiv2,counter))
            
    conj_first_part=[]
    for i in range(2):
        for j in range(1, nclausesDiv2+1):
            if i == 0:
                conj_first_part.append([-x for x in pos_clause[j]] + [v[i][j]])
                for x in pos_clause[j]:
                    conj_first_part.append([-v[i][j], x])
            else:
                conj_first_part.append([-x for x in neg_clause[j]] + [v[i][j]])
                for x in neg_clause[j]:
                    conj_first_part.append([-v[i][j], x])

    conj_s_part=[]
    for i in range(2):
        conj_s_part = conj_s_part + (seq_counter(v[i],r[i],nclausesDiv2))
        
        

    conj_t_part=[]   
    for j in range(1, nclausesDiv2+1):
        conj_t_part.append([r[0][nclausesDiv2][j], o[j]])
        conj_t_part.append([-r[1][nclausesDiv2][j], o[j]])
        conj_t_part.append([-r[0][nclausesDiv2][j], r[1][nclausesDiv2][j], -o[j]])
        #conj_t_part.append([r[0][nclausesDiv2][j], o[j]])
        #conj_t_part.append([-r[1][nclausesDiv2][j], o[j]])
        #conj_t_part.append([-r[0][nclausesDiv2][j], r[1][nclausesDiv2][j], -o[j]])
          
    conj_f_part=[]
    for j in range(1, nclausesDiv2+1):   
        conj_f_part.append([-o_final[0], o[j]])
    o.pop(0)
    conj_f_part.append([-x for x in o]  +  [o_final[0]])

    d =(conj_first_part+conj_s_part+conj_t_part+conj_f_part)

    return d

def not_robust(m, x, x_input, label_x, p,counter,o):
    p = p + 1
    v = len(x_input) 
    n = False
    l = [None] + define_variables(v,counter)
    runtimes = 0

        
    t = [[0]]
    for i in range(1, v+1):
        t.append([0])
        t[i] = t[i] + (define_variables(p,counter))

    forlist  = seq_counter(l,t,p) + [[-t[v][p]]]

    for i in range(1, v+1):
        if x_input[i-1] == 0:
            forlist = forlist + [[-x[i],l[i]]]  
            forlist = forlist + [[x[i],-l[i]]]  
        else:  
            forlist = forlist + [[-x[i],-l[i]]]
            forlist = forlist + [[x[i],l[i]]]
            
    forlist = forlist + m
    
    if not label_x:
        forlist = forlist + [[o[0]]]
    else:
        forlist = forlist + [[-o[0]]]

    start = timeit.default_timer()
    n = is_satisfiable(forlist)
    runtimes = runtimes+timeit.default_timer()-start   

    return runtimes,n


def not_similar(m, m2, x, x_input, label_x, p, counter, o, o2):
    p = p + 1
    v = len(x_input) 
    n = False
    l = [None] + define_variables(v,counter)
    runtimes = 0
    t = [[0]]
    for i in range(1, v+1):
        t.append([0])
        t[i] = t[i] + (define_variables(p,counter))

    seq = seq_counter(l,t,p) 
    forlist  = seq_counter(l,t,p) + [[-t[v][p]]]
    
    for i in range(1, v+1):
        if x_input[i-1] == 0:
            forlist = forlist + [[-x[i],l[i]]]  
            forlist = forlist + [[x[i],-l[i]]]  
        else:  
            forlist = forlist + [[-x[i],-l[i]]]
            forlist = forlist + [[x[i],l[i]]]
    
    forlist = forlist + m + m2  + [[o[0], o2[0]]] + [[-o[0],-o2[0]]]

    start = timeit.default_timer()
    n = is_satisfiable(forlist)
    runtimes = runtimes+timeit.default_timer()-start   
    
    return runtimes,n

def check_similarity(p, encoded, encoded2, x, example, label, counter, o, o2):
    start = timeit.default_timer()
    runtimes, n = not_similar(encoded, encoded2, x, example, label, p, counter, o, o2)
    runtimef = timeit.default_timer()-start
 
    return runtimes, runtimef, not n

 
def check_robustness(p,x,encoded,example,label,counter,o):
    start = timeit.default_timer()
    runtimes, n = not_robust(encoded, x, example,label, p,counter,o)
    runtimef = timeit.default_timer()-start
    if (runtimes > 300):
        n = True
    return runtimes, runtimef, not n