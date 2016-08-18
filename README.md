### Overview

The **ldbod** package provides flexible functions for computing local density-based outlier scores using efficient nearest neighbor search. Functions included in this package can be used for computing local density-based outlier scores in the unsupervised and semi-supervised setting. Two functions are included 'ldbod()' and 'ldbod2()'. Function 'ldbod(X,k,...)' computes outlier scores referencing random subamples of the input data, X. Fucntion 'ldbod2(X,Y,k,...)' computes outlier scores based on a reference data set, Y. Y can be a set of "normal" data points for semi-supervised outlier detection.

Efficient nearest neighbor search is implemented using a k-d tree with default set to exact search. Other options exist for computing kNNs. Refer to **RANN** package for more details on changing parameters for nearest neighbor.

### Installation

To install the package in R use the following command:

``` r
install.packages("devtools")
devtools::install_github("kwilliams83/ldbod")
```

### Examples

To compute

<!-- README.md is generated from README.Rmd. Please edit that file -->
