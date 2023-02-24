
# deepregression Torch (Proof-of-Concept)


This is a proof-of-concept which was done in *Applied Deep Learning*. It shows that torch can be used as well as tensorflow to build and fit a deepregression model from the [deepregression](https://github.com/davidruegamer/deepregression) package.

# Installation

To install the package, use the following command:
``` r
devtools::install_github("marquach/deepregression")
```
Note that the installation requires additional packages (see below) and their installation is currently forced by `deepregression`.

# Requirements

The requirements are given in the `DESCRIPTION`. If you load the package manually using `devtools::load_all`, make sure the following packages are availabe:

  - Matrix
  - dplyr
  - mgcv
  - torch

# Related literature

The following works are based on the ideas implemented in this package:

* [Original Semi-Structured Deep Distributional Regression Proposal](https://arxiv.org/abs/2002.05777)
* [Neural Mixture Distributional Regression](https://arxiv.org/abs/2010.06889)
* [Deep Conditional Transformation Models](https://arxiv.org/abs/2010.07860)
* [Semi-Structured Deep Piecewise Exponential Models](https://arxiv.org/abs/2011.05824)
* [Combining Graph Neural Networks and Spatio-temporal Disease Models to Predict COVID-19 Cases in Germany](https://arxiv.org/abs/2101.00661)

# People that contributed

Many thanks to following people for helpful comments, issues, suggestions for improvements and discussions: 

* Andreas Bender
* Christina Bukas
* Oliver Duerr
* Patrick Kaiser
* Nadja Klein
* Philipp Kopper
* Christian Mueller
* Julian Raith
* Fabian Scheipl
* Matthias Schmid
* Max Schneider
* Ruolin Shen
* Almond Stoecker
* Dominik Thalmeier
* Viet Tran
* Kang Yang
