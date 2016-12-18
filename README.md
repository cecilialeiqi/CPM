# CPM

## Introduction
We proposed two methods for computing dominant eigenvector of a given matrix/graph. Coordinate-wise Power Method (CPM) is for general matrices, and Symmetric Greedy Coordinate Descent (SGCD) is for symmetric matrices [1]. This implementation includes the two proposed methods as well as the traditional power method, Lanczos method with early termination, and VRPCA [2] on dense and synthetic dataset.

## Usage
* Install Eigen (http://eigen.tuxfamily.org/index.php?title=Main_Page#Download) and put it in ./lib/
* make dense
* ./dense (matrix size) (condition number)

## Reference
[1] Lei, Qi, Kai Zhong, and Inderjit S. Dhillon. "Coordinate-wise Power Method." Advances in Neural Information Processing Systems. 2016.

[2] Shamir, Ohad. "A stochastic PCA and SVD algorithm with an exponential convergence rate." Proc. of the 32st Int. Conf. Machine Learning (ICML 2015). 2015.

