# ThreSPCA
A Fast, Provably Accurate Approximation Algorithm for Sparse Principal Component Analysis 

## Description 
We present **ThresPCA**, a provably accurate algorithm based on thresholding the Singular Value Decomposition for the Sparse PCA problem, without imposing any restrictive assumptions on the input covariance matrix. Our thresholding algorithm is conceptually simple; much faster than current state-of-the-art; and performs well in practice.

## Getting Started

### Dependencies

* This package was developed using Python 3.
* It requires some package dependencies listed here:
```
1. Numpy
2. Pandas 
3. Scipy
4. Sklearn  
```
### Installing
Clone the repository with `git clone https://github.com/aritra90/ThresPCA`

### Executing programs
Run the program as

```
sparse_pc, nnz = ThreSPCA(data_matrix, s, l)
```
where `data_matrix` is a matrix with `m` observations in rows and `n` features in columns. `s` is the scalar denoting target sparsity and `l` is the thresholding scalar which we set to *one*. This returns the sparse PC (a vector of order `n`) and the number of non-zero entries `nnz`. 

## Notes
This will ideally give the top PC (PC1), to get other PCs, run this function on `data_matrix_new = data_matrix - data_matrix*vv^T`, where `v` is the top right singular vector of the data matrix. 
Keep iterating on this for obtaining other PCs such as PC2, PC3, PC4, etc. 

## Authors and Correspondence 

Agniva Chowdhury (chowdhu5 at purdue dot edu) 

Aritra Bose (a dot bose at ibm dot com)
