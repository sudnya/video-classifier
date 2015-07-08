# **Lucious Technical Documentation**

## Overview

Lucious is a high-performance open-source neural network classification system.
It provides fast implementations of recently published research techniques drawn
from the fields of computer vision, machine learning, and optimization with clean
software interfaces that builds on multiple platforms with minimal dependencies.

This project implements a general purpose classification system.  The focus of
the project is on achieving robust performance (classification accuracy) with
minimal human interaction. Unlike most classification systems, which rely on
domain specific heuristics to perform feature selection, Lucious uses Deep Neural
networks to automatically learn important features automatically over time without
any human interaction.

## 1. Deep Learning

'Deep Learning' is used to describe the process of
automatically discovering hierarchical features in large data sets without human interaction.
Without deep learning, it is necessary to manually design features that are
robust to variations in color, alignment, and noise.

Many projects rely
on building domain-specific feature selection systems that preprocess the
data and produce a set of features or metrics that attempt to capture the
essential information in the input data set using a significantly smaller
representation.  Feature selection is effective at reducing the dependence on
labeled data because it simplifies the problem being presented to the supervised
learning system (i.e. determine the class of a dataset using only the most
relevant information rather than all information), and Sift is an example
of a widely used feature selection system.  Despite their usefulness,
manually-crafted feature selection systems have a fatal flaw; they are designed
by developers and tailored to specific problems.  As a consequence, they
are often brittle (because they are designed using developer intuition),
and require a complete redesign when moving from one type of input data to
another (e.g. from video to text).

Deep learning attempts to
address the shortcomings of feature selection systems by providing a framework
that generates them automatically.  At a high level, it works using an
artificial neural network with multiple levels of hierarchy. After training, the inner layers
of this network can be used directly as features; in other words, the network
itself becomes a feature selection system.

The following image includes a visualization of some of the low level features
(inputs that a pooling layer neuron maximally responds to) that were learned by Lucious after
being presented 10,000 random images.

![Layer One Features](/documentation/tech-report/images/first-layer-neurons-8x8-tiles.jpg "Layer One Feature Responses")

These compare favorably to Gabor Filter function responses in the top-right of the following figure, which have been found
in the first layers of the visual cortex in mammalians, and perform well at visual classification tasks:

![Layer One Features](/documentation/tech-report/images/SoftFeatures.png "Gabor Function Responses and Spatial Domain Representation")

From this paper:
>    Stanton R. Price ; Derek T. Anderson ; Robert H. Luke ; Kevin Stone ; James M. Keller;
>    Automatic detection system for buried explosive hazards in FL-LWIR based on soft feature
>    extraction using a bank of Gabor energy filters. Proc. SPIE 8709, Detection and Sensing of
>    Mines, Explosive Objects, and Obscured Targets XVIII, 87091B (June 7, 2013);

The accompanying figure on the top-right shows the corresponding spatial domain representation (impulse response) of each of the Gabor Filters.

Lucious is influenced by the following research groups in deep learning:

* Andrew Ng (Automatic Feature Selection): http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial
* Yann Lecun (Learning Feature Hierarchies): http://www.cs.nyu.edu/~yann/talks/lecun-20110505-lip6.pdf
    * Convolutional and Pooling Networks: http://deeplearning.net/tutorial/lenet.html

## 2. Supervised Learning

Lucious uses a well known design involving an artificial neural network for supervised learning.  The input
data is preprocessed using the feature selection network.

## 3. Classification

Once a network has been trained, classification is performed simply by
propagating the input data through the feature selection and classification
neural networks to obtain a predicted class.

## 4. High Performance Architecture

Lucious is designed with high performance in mind to be able to scale to large
data sets. It uses three principles to achieve high performance.

* Use high performance accelerator architectures (e.g. GPUs) in single nodes.
* Use a scalable design that can leverge distributed systems.
* Design convolutional neural networks in the first few layers to reduce the computational complexity of large networks.

A long term vision for this project is to scale to millions of input features
on single node systems (handle mega-pixel images without any downsampling), and
billions of input features on thousand node clusters.

# System
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

In Lucious, the classification system is called a Model.  Models contain a serialized representation
of the entire system, including all neural networks and associated metadata.  The model builder is
used to create a blank model with new parameters (network topology, labels, etc).

Once a model has been created, it can be updated with supervised or unsupervised learning, or it
can be used directly to perform classification.  Lucious currently supports video and image input
data sets, but none of the modules are tied to this representation.  Supporting a new input format
(e.g. text or numeric data) would simply require a new data conversion component that transforms the input
into a standardized vector of floating point values, and presents it to the existing models.

The details of the libraries and modules are described next.

### The Neural Network Library

 Each neural network contains multiple layers. Each layer is represented as a collection of sparse matrices.  The neural network library starts by initializing the network randomly. The inputs to the neural network are pixels of a down-sampled image in the form of a matrix. With pixel values as input to the neural network, we use the optimization library to train the neural network. The neural network is trained until the cost function for the neural network is minimized. The cost function captures the difference between the expected output and the predictions made by the neural network.

### The optimization library

 The optimization library aims to reduce the difference between the actual output (from the labeled data) and the output predicted by the neural network by modifying individual neuron weights. This library contains multiple implementations.
a.) The nesterov stochastic gradient descent with momentum algorithm.
b.) The Broyden–Fletcher–Goldfarb–Shanno algorithm (an approximation of Newtons method for unconstrained linear optimization).

### The linear algebra library

 The linear algebra library leverages the optimized Matrix operations from pre-existing implementations (ATLAS or CUBLAS). The smallest unit for calculations in the neural network & optimizer is the Matrix. The linear algebra library translates these into calls to the Matrix library.

### The video library

 OpenCV is used for all the image processing required in this project. The input video is converted into a series of images. These images are then converted to a lower resolution (using tiling) and then finally to a matrix of pixel values.  Pixel values are standardized and then passed, and then can be passed directly into a network. The resolution chosen depends on the size of the neural network.  For a sufficiently large network, no downsampling is performed and the network is connected directly to raw pixel values.

### The model builder

 At each of the following steps, the neural network generated may be written out to the disk. The network is a represented as a model with various attributes and their corresponding values. This model is then serialized and written to a compressed file. Writing these models decouples each step and thus allows the capability of resuming with the help of a model file. Eg: The unsupervised learning step takes many hours of running video to automatically generate a feature selector neural network. This network can be saved to a file and then reused with different sets of training data to create a classification neural network. This saves the time required to rerun the unsupervised learning step.

### The unsupervised learning module

 This is the first step. We start by creating a new model. This new model is simply a randomly initialized neural network. We then feed video database to this network. After processing sufficiently large amounts of input images the neural network automatically configures itself to be a feature selector. Here are some details of how the neural network is configured:
The idea behind a sparse autoencoder is that features in some data can be attributed to only some relevant parts of the input space, rest of it is noise. Using a steady reduction factor, the size of the neural network layers is decreased significantly. At this stage, we start adding new layers one at a time and train all the neurons in the new layer by running back-propagation algorithm.

### The supervised learning module

 This module uses the neural network from the earlier unsupervised learning stage as the starting point (instead of a randomly initialized neural network). At the start, this neural network is capable to selecting features. We feed in labeled data to this network to generate a classifier neural network. This is the training step which uses labeled data with gesture names for images or video frames. At the end of this step i.e. once the neural network output is within threshold of the expected output, it is written out as a classifier neural network ready to be used for testing with images.

### The classification module

 This module uses both the neural network (the feature selector from the unsupervised learning step) and the classifier neural network (from the supervised learning step) to perform classification in test images.  It is really a convenience interface that abstracts some of the low level operations involved in setting up the input data and training the networks contained in a model.  It presents a clean interface for feeding input labeled or unlabeled data to the model.  Operations like random sampling and input caching are automated using the classification module framework.

