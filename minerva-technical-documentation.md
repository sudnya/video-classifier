# **Minerva Technical Documentation**

## Overview

This project implements an unsupervised learning technique that uses a neural network and processes multiple hours of video as input. This unsupervised learning technique does automatic feature detection & generates a feature selector neural network. 
The feature selector neural network is then trained using labeled images to obtain a neural network capable of classification. The labeled dataset used for training was obtained from http://www.kaggle.com/c/multi-modal-gesture-recognition

## 1. Unsupervised Learning
Automatic feature selection or 'deep learning' is a problem that has recently received a lot of attention. As the amount of data available to run the Machine Learning algorithm grows constantly, this problem of manually labeling input data becomes more acute. Thus, the need for systems that can automatically detect features becomes more apparent. This project accomplishes 'deep learning' a.k.a 'automatic feature selection' by implementing the Sparse Autoencoder described in http://stanford.edu/class/cs294a/sparseAutoencoder.pdf The main idea .......

## 2. Supervised Learning

## 3. Classification

## System
The system is composed of the following main components:
1.  The artificial neural network
2.  The optimization library
3.  The linear algebra library
4.  The video library


 Each neural network contains multiple layers. Each layer is represented as a collection of sparse matrices.  The neural network library starts by initializing the network randomly. The inputs to the neural network are pixels of a down-sampled image in the form of a matrix. We use the back-propagation 

2. The optimization library
3. The matrix library
4. Tools (to read in video, images & convert them into Matrix)


