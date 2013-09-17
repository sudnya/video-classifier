# **Minerva Technical Documentation**

## Overview

This project implements an unsupervised learning technique that uses a neural network and processes multiple hours of video as input. This unsupervised learning technique does automatic feature detection & generates a feature selector neural network. 
The feature selector neural network is then trained using labeled images to obtain a neural network capable of classification. The labeled dataset used for training was obtained from http://www.kaggle.com/c/multi-modal-gesture-recognition

## 1. Unsupervised Learning
Automatic feature selection or 'deep learning' is a problem that has recently received a lot of attention. As the amount of data available to run the Machine Learning algorithm grows constantly, this problem of manually labeling input data becomes more acute. Thus, the need for systems that can automatically detect features becomes more apparent. This project accomplishes 'deep learning' a.k.a 'automatic feature selection' by implementing the Sparse Autoencoder described in http://stanford.edu/class/cs294a/sparseAutoencoder.pdf The main idea .......

## 2. Supervised Learning

## 3. Classification

## System
The system is composed of the following main component libraries:

* The artificial neural network
* The optimization library
* The linear algebra library
* The video library

These libraries are used internally by the following interfaces:

* The model builder
* The supervised learning module
* The unsupervised learning module
* The classification module

In Minerva, the classification system is called a Model.  Models contain a serialized representation
of the entire system, including all neural networks and associated metadata.  The model builder is used to create a blank model with new parameters (network topology, labels, etc).

Once a model has been created, it can be updated with supervised or unsupervised learning, or it can be used directly to perform classification.  Minerva currently supports video and image input data sets, but none of the modules are tied to this representation.  Supporting a new input format (e.g. text or numeric data) would simply require a new driver application that converts the input into a standardized vector of floating point values, and presents it to the existing models.

The details of the libraries and modules are described next.

### The Neural Network Library

 Each neural network contains multiple layers. Each layer is represented as a collection of sparse matrices.  The neural network library starts by initializing the network randomly. The inputs to the neural network are pixels of a down-sampled image in the form of a matrix. With pixel values as input to the neural network, we use the back-propagation algorithm to train the neural network. The neural network is trained until the cost function for the neural network is minimized. The optimization library is used to compute this cost function. 

### The optimization library

 The optimization library aims to reduce the difference between the actual output (from the labeled data) and the output predicted by the neural network. This library contains a couple of implementations. 
a.) Gradient descent with linear simulated annealing
b.) The Multilevel optimizer uses a greedy heurisitc with simulated annealing & local search (with tabu search)

### The linear algebra library

 The linear algebra library leverages the optimized Matrix operations from pre-existing implementations. The smallest unit for calculations in the neural network & optimizer is the Matrix. The linear algebra library translates these into calls to the Matrix library.

### The video library

 OpenCV is used for all the image processing required in this project. The input video is converted into a series of images. These images are then converted to a lower resolution & then finally to a matrix of pixel values. 

### The model builder

 At each of the following steps, the neural network generated is written out to the disk. The network is a represented as a model with various attributes & their corresponding values. This model is then serialized & written to a file.
 Writing these models decouples each step & thus allows the capability of resuming with the help of a model file. Eg: The unsupervised learning step takes many hours of running video to automatically generate a feature selector neural network. This network can be saved to a file & then reused with different sets of training data to create a classification neural network. This saves the time required to rerun the unsupervised learning step.

### The supervised learning module
### The unsupervised learning module
### The classification module



