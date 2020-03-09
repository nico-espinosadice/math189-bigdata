# EM Paper Review
Nico Espinosa Dice

March 9, 2020

## Expectation-Maximization for Learning Determinantal Point Processes
### Authors: 
Jennifer Gillenwater, Alex Kulesza, Emily Fox, Ben Taskar

### Link:
https://arxiv.org/pdf/1411.1088v1.pdf

### Summary
In this paper, the authors present a method of learning the full kernel matrix for a determinantal point process. 

("A determinantal point process (DPP) is a probabilistic model of set diversity compactly parameterized by a positive semi-definite kernel matrix" – Gillenwater, Kulesza, Fox, Taskar 2014). 

By learning the entires of the kernel matrix, they are able to utilize the DPP for numerous applications. Previously, this problem required the learning of the kernel matrix's entries by maximizing the log-likelihood of the available data, which the authors suggest was NP-hard. 


Instead, to accomplish the task with their new method, they first parameterized the kernel of the DPP with eigenvectors and eigenvalues, instead of to simply matrix entries. Next, they used the expectation-maximization algorithm to find the maximum likelihood efficiently.

When the new algorithm was tested on a real application, it did significantly better than maximizing the log-likelihood using the gradient descent method.

### Algorithm
I was not able to find an implementation of code for this paper, but they did provide the algorithm in pseudocode, which I have pictured below.

(Gillenwater, Kulesza, Fox, Taskar 2014):
