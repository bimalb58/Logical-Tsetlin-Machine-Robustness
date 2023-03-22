This document is a how-to for running the experiments detailed in the paper.
Download MNIST-c and IMDB datasets as per instructions below.

================
Requirements
================
Python 3.7.x, https://www.python.org/
Numpy, http://www.numpy.org/
PyCUDA, https://documen.tician.de/pycuda/
Scikit-learn, https://scikit-learn.org/
Keras, https://keras.io/
Tensorflow, https://www.tensorflow.org/install
_______________________________________________________________

================
Installation
================

pip install PyTsetlinMachineCUDA
_______________________________________________________________

========
Datasets
========
MNIST-C: https://zenodo.org/record/3239543#.ZBsnFnbMJNM
IMDB: Can be downloaded from Keras from imdb_preprocess.py

________________________________________

================
Preprocessing
================
The script produces .pkl file, which is later used in robustness and similarity check.

$python imdb_preprocess.py 
$python MNIST_preprocess.py 

================
Robustness
================

$python Robustness_check.py 

================
Similarity
================

$python similarity_check.py 

----FIN----