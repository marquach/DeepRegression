
# deepregression Torch (Masterthesis)
## Content

By combining additive models and neural networks, it becomes possible to expand the scope of statistical regression and extend deep learning-based approaches using interpretable structured additive predictors. The content of this thesis is the implementation of semi-structured regression, a general framework that allows the combination of many different structured additive models and deep neural networks, using the R-native deep learning language torch. Using existing functionalities in the TensorFlow-based software R package deepregression as a blueprint, this thesis will implement torch-based alternatives to existing functions that, e.g., allow to 

- extract estimated coefficients of linear predictors; (DONE)
- plot estimated non-linear functions; (DONE)
- predict on unseen data; (DONE)
- perform cross-validation; (DONE)
- specify a variety of different distribution for distribution learning; (DONE)
- compute deep ensembles of semi-structured regression models; (DONE)
- specify models for and learn with multi-modal data sets (e.g., image data combined with tabular data). (DONE)

The thesis will demonstrate the efficacy of this framework and compare it to existing software implementation deepregression through numerical experiments, while also highlighting its special merits in a real-world application on the airbnb dataset. 


# Stucture

`scripts` contains all of the scripts, which were created in the process of the masterthesis.

`scripts\Comprehension_scripts` contains the scripts that were used to get in touch with: 

 
 `scripts\milestones_presentation` lorem ipsum


`scripts\deepregression_functions.R` contains the $\texttt{tensorflow} \Rightarrow \texttt{torch}$  *translated* deepregression functions, which were needed to get the proof-of-concept work. 
Most of these functions were named like the ones from the [deepregression](https://github.com/neural-structured-additive-learning/deepregression) package, but a `_torch` was added to the name.

Also some showcases (`showcases`) and tests (`tests`) were included to check whether the implementation works.

An intercept-only, a linear model, an additive model, a structured model (combination of all mentioned before), a deep model and a semi-structured model were used as examples. 

# Outlook:

# Installation (does not work because private)

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
