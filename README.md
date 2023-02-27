
# deepregression Torch (Proof-of-Concept)

This project was part of the Winter 2022/23 course Applied Deep Learning with TensorFlow and Pytorch at the LMU Munich, supervised by Dr. David Rügamer and Chris Kolb.

The project is a proof-of-concept. It shows that torch can be used as well as tensorflow to build and fit a deepregression model from the [deepregression](https://github.com/davidruegamer/deepregression) package.
deepregression’s core functionality is to define and fit (deep) distributional regression models. 

As this is just a proof-of-concept not all functionalities are already implemented.
Up to this time-point (02.27.2023) only the processors for the intercept-layer, linear-layer and the gam-layer are implemented for the torch approach.
Also the orthogonalization is not implemented. The orthogonalization ensures identifiability of the structured term(s). 
Simply put, the orthogonalization constrains the deep network such that its latent learned features cannot
represent feature effects that already exist in the structured predictor.

# Also to mention

Some things still need some dedication:

  - There is no summary of the model yet, which is comparable to the Keras summary 
    - it looks like there is also no really summary in python itself [(Link)](https://github.com/TylerYep/torchinfo)
  - Also, the layers of the distribution parameters are not yet named. Same holds for the different layers which represent the parts of used formula. 
    - there is already an idea, which uses a combination of eval() and parse()
  - Also, there are no generic fit(), predict(), plot() and coef() for the torch model
    - Doable but not needed for the proof-of-concept

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
  - luz

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
