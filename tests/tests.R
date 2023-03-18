#  tests can be found here

library(testthat)
library(deepregression)
library(torch)
library(luz)
devtools::load_all("deepregression-main/")
source('Scripts/deepregression_functions.R')
source("tests/data_4_tests.R")
# Dann müssen Modelle nicht mehr gefittet werden
# aber dauert lange, besser anders lösen
source("showcases/showcases.R")


orthog_options = orthog_control(orthogonalize = F)

test_that("Throws error when family is implemented and if orthogonalizations is used",{
  expect_error(deepregression(
    list_of_formulas = list(loc = ~ 1, scale = ~ 1),
    data = toy_data, y = toy_data$y,
    orthog_options = orthog_options, return_prepoc = F, 
    subnetwork_builder = subnetwork_init_torch,
    model_builder = torch_dr,
    engine = "torch", family = "gamma"),
    "Only normal distribution tested")
  
  expect_error(deepregression(
    list_of_formulas = list(loc = ~ 1, scale = ~ 1),
    data = toy_data, y = toy_data$y, return_prepoc = F, 
    subnetwork_builder = subnetwork_init_torch,
    model_builder = torch_dr,
    engine = "torch"),
    "Orthogonalization not implemented for torch")
  })


formula <- ~ -1 + deep_model(x1,x2,x3)

nn_tf <- function(x) x %>%
  layer_dense(units = 32, activation = "relu", use_bias = FALSE) %>%
  layer_dense(units = 1, activation = "linear")

nn_torch <- function() nn_sequential(
  nn_linear(in_features = 3, out_features = 32, bias = F),
  nn_relu(),
  nn_linear(in_features = 32, out_features = 1)
)

test_that("Used Formulas are same and number of # of parameters is  same",{
deep_model_tf <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y,orthog_options = orthog_options,
  list_of_deep_models = list(deep_model = nn_tf), engine = "tf"
)

deep_model_torch <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y, orthog_options = orthog_options,
  list_of_deep_models = list(deep_model = nn_torch),
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  engine = "torch"
)

expect_true(all.equal(deep_model_tf$init_params$list_of_formulas,
          deep_model_torch$init_params$list_of_formulas))

expect_true(all.equal(deep_model_tf$model$count_params(),
Reduce(x = (lapply(deep_model_torch$model()$parameters, length)), sum)))
})

