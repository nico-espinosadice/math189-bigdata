# EM Paper Review
Nico Espinosa Dice

March 9, 2020

## Paper: Expectation-Maximization for Learning Determinantal Point Processes (2014)
### Authors: 
Jennifer Gillenwater, Alex Kulesza, Emily Fox, and Ben Taskar

### Link:
Paper: https://arxiv.org/pdf/1411.1088v1.pdf

Code: https://code.google.com/archive/p/em-for-dpps/source/default/source

### Summary
In this paper, the authors present a method of learning the full kernel matrix for a determinantal point process. 

("A determinantal point process (DPP) is a probabilistic model of set diversity compactly parameterized by a positive semi-definite kernel matrix" – Gillenwater, Kulesza, Fox, Taskar 2014). 

By learning the entires of the kernel matrix, they are able to utilize the DPP for numerous applications. Previously, this problem required the learning of the kernel matrix's entries by maximizing the log-likelihood of the available data, which the authors suggest was NP-hard. 


Instead, to accomplish the task with their new method, they first parameterized the kernel of the DPP with eigenvectors and eigenvalues, instead of to simply matrix entries. Next, they used the expectation-maximization algorithm to find the maximum likelihood efficiently.

When the new algorithm was tested on a real application, it did significantly better than maximizing the log-likelihood using the gradient descent method.

### Algorithm and Code
They implemented the algorithm for a recommendation system. Unfortunately, it is written in Matlab, so I was not able to get it to run on my computer. (The link to the code is given above). However, the algorithm is presented in pseudocode below.

(Gillenwater, Kulesza, Fox, Taskar 2014):
![Algorithm from Paper](https://github.com/nico-espinosadice/math189-bigdata/blob/master/Homework6/EM_Paper_Algorithm.png)
