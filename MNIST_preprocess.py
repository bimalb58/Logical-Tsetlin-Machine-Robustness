from PyTsetlinMachineCUDA.tm import TsetlinMachine, MultiClassTsetlinMachine
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

data_dir = r'../mnist_c'  #Change the data_dir to where you extracted the mnist_c.zip

_TEST_IMAGES_FILENAME = 'test_images.npy'
_TEST_LABELS_FILENAME = 'test_labels.npy'

#Simply comment out the corruptions that you don't want to use. Remember to keep the 'identity' as it is the actual test_images.
CORRUPTIONS = [
    'identity',
    'shot_noise',
    'impulse_noise',
    'glass_blur',
    'motion_blur',
    'shear',
    'scale',
    'rotate',
    'brightness',
    'translate',
    'fog',
    'stripe',
    'spatter',
    'dotted_line',
    'zigzag',
    'canny_edges',
]

# Change esp to change the probability of getting a corrupted image during testing.
esp = 1
number_of_features = 28*28
n_clauses = 1000
epochs = 100

(X_train, Y_train), (_, _) = mnist.load_data()
all_test_images = []
all_test_labels = []

for corruption in CORRUPTIONS:
    images_file = os.path.join(data_dir, corruption, _TEST_IMAGES_FILENAME)
    labels_file = os.path.join(data_dir, corruption, _TEST_LABELS_FILENAME)
    images = np.load(images_file)
    labels = np.load(labels_file)
    all_test_images.append(images)
    all_test_labels.append(labels)

all_test_images = np.array(all_test_images)
all_test_labels = np.array(all_test_labels)

print(all_test_images.shape, all_test_labels.shape)

X_test_corrupted = []
Y_test_corrupted = []
X_test = []
Y_test = []

for i in range(10000):
    corrupt = np.random.choice(range(len(CORRUPTIONS)),p=[1-esp]+[esp/(len(CORRUPTIONS)-1)]*(len(CORRUPTIONS)-1))
    X_test_corrupted.append(all_test_images[corrupt][i])
    Y_test_corrupted.append(all_test_labels[corrupt][i])
    X_test.append(all_test_images[0][i])
    Y_test.append(all_test_labels[0][i])
X_test_corrupted = np.array(X_test_corrupted)
Y_test_corrupted = np.array(Y_test_corrupted)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
print(X_test_corrupted.shape, Y_test_corrupted.shape)
print(X_test.shape, Y_test.shape)

# change to white/black pixels
X_train = np.where(X_train >= 75, 1, 0) 
X_test_corrupted = np.where(X_test_corrupted >= 150, 1, 0)
X_test = np.where(X_test >= 150, 1, 0)

num = 10
num_row = 2
num_col = 5


# plot images
print("Corrupted test images")
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(X_test_corrupted[i], cmap='gray')
    ax.set_title('Label: {}'.format(Y_test_corrupted[i]))
plt.tight_layout()
plt.show()

print("Original test images")
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(X_test[i], cmap='gray')
    ax.set_title('Label: {}'.format(Y_test[i]))
plt.tight_layout()
plt.show()

# get number of changes pixels
for i in range(num):
    print("Number of flips in the image %d: %d" % (i+1, np.sum(X_test_corrupted[i] != X_test[i])))

X_train = np.reshape(X_train,(X_train.shape[0],number_of_features))
X_test = np.reshape(X_test,(X_test.shape[0],number_of_features))
X_test_corrupted = np.reshape(X_test_corrupted,(X_test_corrupted.shape[0],number_of_features))

tm = MultiClassTsetlinMachine(n_clauses, 25, 10.0, boost_true_positive_feedback=1)

max = 0.0

for i in range(epochs):
    tm.fit(X_train, Y_train, epochs=1, incremental=True)

    result_test = 100*(tm.predict(X_test) == Y_test).mean()
    result_train = 100*(tm.predict(X_train) == Y_train).mean()
        
    if result_test > max:
        max = result_test

    print("#%d Train Accuracy: %.2f,  Test Accuracy: %.2f\n" % (i, result_train, result_test))

import pickle 
f_tm = open("TM_MNIST.pickle", "wb+")
pickle.dump(tm, f_tm)
f_tm.close()

f_tm = open("MNIST_test.pickle", "wb+")
pickle.dump(X_test, f_tm)
f_tm.close()
