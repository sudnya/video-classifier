# **Minerva Technical Documentation**

## Overview

This project implements a general purpose classification system.  The focus of
the project is on achieving robust performance (classification accuracy) with
minimal human interaction. Unlike most classification systems, which rely on
domain specific heuristics to perform feature selection, Minerva uses an
unsupervised learning technique called 'sparse autoencoding' that learns
important features automatically over time without any human interaction.
The sparse autoencoder is implemented with a convolutional neural network that takes raw
data as input and produces a set of features that attempt to capture the
essential information in that data.  It is trained by streaming through massive
unlabeled input data sets. Once the sparse autoencoder has learned useful
features, it is connected to a more traditional classification system that
is trained using supervised learning.  This system is also implemented with
a neural network, and it attempts to discover complex relationships between
generated features and output classes.  

Although Minerva is designed the handle arbitrary input data, this project
includes additional supporting modules for performing classification on video
data. As our first case study, we are using Minerva to perform automatic gesture
recognition.  

The labeled dataset used for training was obtained from http://www.kaggle.com/c/multi-modal-gesture-recognition

## 1. Unsupervised Learning

Unsupervised learning (or 'deep learning') is used to describe the process of
automatically discovering patterns in large data sets without human interaction.
Without unsupervised learning, it is necessary to manually label thousands
or millions of video frames to perform classification (identify what an image
contains) that is robust to variations in color, alignment, and noise.  This
is not feasible for most project since developers do not have the resources
to gather such a large number of videos.  

As a response, many projects rely
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

A sparse autoencoder is an unsupervised learning technique that attempts to
address the shortcomings of feature selection systems by providing a framework
that generates them automatically.  At a high level, it works using an
artificial neural network that is trained with unlabaled data and configured
with a very specific topology that guides that inner layers of the network
to respond to patterns in the input data that capture the majority of
information in that data.  

The details of sparse autoencoders are described here
http://stanford.edu/class/cs294a/sparseAutoencoder.pdf .  

## 2. Supervised Learning

Minerva uses a well known neural network for supervised learning.  The input
data is preprocessed using the feature selection network.  

## 3. Classification

Once a network has been trained, classification is performed simply by
propagating the input data through the feature selection and classification
neural networks to obtain a predicted class.

## 4. High Performance Architecture

Minerva is designed with high performance in mind to be able to scale to large
data sets. It uses three principles to achieve high performance.  

* Use high performance accelerator architectures (e.g. GPUs) in single nodes.
* Use a scalable design that can leverge distributed systems.
* Design convolutional neural networks in the first few layers to reduce the computational complexity of large networks.

A long term vision for this project is to scale to millions of input features
on single node systems, and billions of input features on thousand node
clusters.

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

In Minerva, the classification system is called a Model.  Models contain a serialized representation
of the entire system, including all neural networks and associated metadata.  The model builder is used to create a blank model with new parameters (network topology, labels, etc).

Once a model has been created, it can be updated with supervised or unsupervised learning, or it can be used directly to perform classification.  Minerva currently supports video and image input data sets, but none of the modules are tied to this representation.  Supporting a new input format (e.g. text or numeric data) would simply require a new driver application that converts the input into a standardized vector of floating point values, and presents it to the existing models.

The details of the libraries and modules are described next.

### The Neural Network Library

 Each neural network contains multiple layers. Each layer is represented as a collection of sparse matrices.  The neural network library starts by initializing the network randomly. The inputs to the neural network are pixels of a down-sampled image in the form of a matrix. We use the back-propagation 

### The optimization library
### The linear algebra library
### The video library

### The model builder
### The supervised learning module
### The unsupervised learning module
### The classification module



