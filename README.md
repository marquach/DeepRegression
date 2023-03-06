
# deepregression Torch (Proof-of-Concept)

This [project](https://docs.google.com/presentation/d/12HXZZBmmlvctInJTBOdDjULu1Eur97WB7ucGwSFCMaA/mobilepresent?slide=id.p) was part of the Winter 2022/23 course Applied Deep Learning with TensorFlow and Pytorch at the LMU Munich, supervised by [Dr. David Rügamer](https://www.slds.stat.uni-muenchen.de/people/ruegamer/) and [Chris Kolb](https://www.slds.stat.uni-muenchen.de/people/kolb/).

The project is a proof-of-concept. It shows that $\texttt{torch}$ can be used as well as $\texttt{tensorflow}$ to build and fit a deepregression model from the [deepregression](https://github.com/neural-structured-additive-learning/deepregression) package.
deepregression’s core functionality is to define and fit (deep) distributional regression models. 

As this is just a proof-of-concept, not all functionalities are implemented.
Most of the $\texttt{tensorflow} \Rightarrow \texttt{torch}$  *translated* functions will be adapted and improved in the further processes.
Up to this time-point (02.27.2023) only the following processors/layers:

- $\texttt{intercept-layer}$
- $\texttt{linear-layer}$
- $\texttt{gam-layer}$

are implemented for the $\texttt{torch}$ approach.

Also the orthogonalization is not implemented. The orthogonalization ensures identifiability of the structured term(s). Simply put, the orthogonalization constrains the deep network such that its latent learned features cannot
represent feature effects that already exist in the structured predictor.

# Stucture

`scripts` contains all of the scripts, which were created in the process of the project.

`scripts\Comprehension_scripts` contains the scripts that were used to get in touch with: 

 - with the [deepregression](https://github.com/neural-structured-additive-learning/deepregression) package 
 - the deep learning language  [torch](https://cran.r-project.org/web/packages/torch/) and also the high-level API [luz](https://cran.r-project.org/web/packages/luz/)
 - distribution learning
 
 `scripts\milestones_presentation` contains the scripts that were used to do some (1.-4.) of [suggested steps](https://docs.google.com/presentation/d/12HXZZBmmlvctInJTBOdDjULu1Eur97WB7ucGwSFCMaA/mobilepresent?slide=id.g188f35a056e_0_35). 


`scripts\deepregression_functions.R` contains the $\texttt{tensorflow} \Rightarrow \texttt{torch}$  *translated* deepregression functions, which were needed to get the proof-of-concept work. 
Most of these functions were named like the ones from the [deepregression](https://github.com/neural-structured-additive-learning/deepregression) package, but a `_torch` was added to the name.

Also some showcases (`showcases`) and tests (`tests`) were included to check whether the implementation works.

An intercept-only, a linear model, an additive model, a structured model (combination of all mentioned before), a deep model and a semi-structured model were used as examples. 

All of them were build and fitted by deepregression with both engines  $\texttt{tensorflow}$ & $\texttt{torch}$ . The first three models and their combination were also build and fitted with [gamlss](https://www.gamlss.com) or with [mgcv](https://cran.r-project.org/web/packages/mgcv/index.html), as the remaining ones can't be covered by [gamlss](https://www.gamlss.com) or by [mgcv](https://cran.r-project.org/web/packages/mgcv/index.html).

# Outlook:

  - implement torch-based alternatives to existing functions
    - implement remaining layers/processors
    - generic functions like plot, coef, predict, ...
  - paste functions from `deepregression_functions.R` to correct place
  - create more tests and showcases
    - compare performance of both deepregression versions ($\texttt{tensorflow}$ & $\texttt{torch}$)

    
# Installation

To install the package, use the following command:
``` r
devtools::install_github("marquach/deepregression/deepregression-main")
```
Note that the installation requires additional packages (see below).

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

* Dr. David Rügamer
* Chris Kolb
* Nora Valiente Bauer
