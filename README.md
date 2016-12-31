# Coordinate-wise Power Method (CPM)

## Introduction
We proposed two methods for computing dominant eigenvector of a given matrix/graph. Coordinate-wise Power Method (CPM) is for general matrices, and Symmetric Greedy Coordinate Descent (SGCD) is for symmetric matrices [1]. This implementation includes the two proposed methods as well as the traditional power method, Lanczos method with early termination, and VR-PCA [2] on dense and synthetic dataset as well as real and sparse dataset.

## Usage
* Install Eigen (http://eigen.tuxfamily.org/index.php?title=Main_Page#Download) and put it in ./lib/
* make 
* ./dense [matrix size] [condition number]
* ./sparse [./data/filename] [matrix size]

	The data format is required to be "node_i node_j" in each line, indicating that nodes i and j are connected. And the nodes should be labeled from 1 to n, the matrix size.

	After running the code, the output file will be saved in result/output.csv. It shows how the accuracy increases over time for the five methods implemented.

## Reference
[1] Lei, Qi, Kai Zhong, and Inderjit S. Dhillon. "Coordinate-wise Power Method." Advances in Neural Information Processing Systems. 2016.

[2] Shamir, Ohad. "A stochastic PCA and SVD algorithm with an exponential convergence rate." Proc. of the 32st Int. Conf. Machine Learning (ICML 2015). 2015.

