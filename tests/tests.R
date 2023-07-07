#  tests can be found here

library(testthat)
library(deepregression)
library(torch)
library(luz)
devtools::load_all("deepregression-main/")
source('Scripts/deepregression_functions.R')
source("tests/ADL_project/data_4_tests.R")
orthog_options = orthog_control(orthogonalize = F)


# Networks
nn_tf <- function(x) x %>%
  layer_dense(units = 32, activation = "relu", use_bias = FALSE) %>%
  layer_dense(units = 1, activation = "linear")

nn_torch <- function() nn_sequential(
  nn_linear(in_features = 3, out_features = 32, bias = F),
  nn_relu(),
  nn_linear(in_features = 32, out_features = 1)
)



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



formula <- ~ 1 + deep_model(x1,x2,x3) + s(xa) + x1
semi_structured_tf <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y,orthog_options = orthog_options,
  list_of_deep_models = list(deep_model = nn_tf), engine = "tf"
)

semi_structured_torch <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y, orthog_options = orthog_options,
  list_of_deep_models = list(deep_model = nn_torch),
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  engine = "torch"
)


semi_structured_tf %>% fit(epochs = 500, early_stopping = F, batch_size = 32)
semi_structured_torch %>% fit(epochs = 500,
                              early_stopping = F, batch_size = 32)

#Compare generics of both approaches ()
#coef

test_that("coef implementation works",{
expect_equal(
  length(coef(semi_structured_tf, which_param = 1, type = "linear")),
  length(coef(semi_structured_torch, which_param = 1, type = "linear")))
expect_equal(
  length(coef(semi_structured_tf, which_param = 1, type = "smooth")),
  length(coef(semi_structured_torch, which_param = 1, type = "smooth")))
expect_equal(
  length(coef(semi_structured_tf, which_param = 2, type = "linear")),
  length(coef(semi_structured_torch, which_param = 2, type = "linear")))
expect_equal(
  lapply(coef(semi_structured_tf, which_param = 1, type = "smooth"), length),
  lapply(coef(semi_structured_torch, which_param = 1, type = "smooth"), length))
})

expect_equal(
  coef(semi_structured_tf),
  coef(semi_structured_torch),
  tolerance = 0.5)


# train (later) longer to reduce tolerance
test_that("plot implementation works",{
  expect_equal(
    plot(semi_structured_tf, which = "s(xa)"),
   plot(semi_structured_torch, which = "s(xa)"), tolerance = 0.5
   )
  })


#predict works mostly (batch_size not, because needs generator)
test_that("basic predict implementation works (batch_size not)",{
expect_equal(predict(semi_structured_tf),
             predict(semi_structured_torch), tolerance = 0.25)

first_obs <- data[1,]
expect_equal(fitted(semi_structured_torch)[1],
             predict(semi_structured_torch, newdata = first_obs)[1,])

expect_error(predict(semi_structured_torch, batch_size = 32))
})


dist_tf <- semi_structured_tf %>% get_distribution()
dist_torch <- semi_structured_torch %>% get_distribution()
str(dist_tf$tensor_distribution, 1)
str(dist_torch, 1)

dist1_tf <- semi_structured_tf %>% get_distribution(first_obs)
dist1_torch <- semi_structured_torch %>% get_distribution(first_obs)

meanval_tf <- semi_structured_tf %>% mean(first_obs) %>% c
meanval_torch <- semi_structured_torch %>% mean(first_obs) %>% c

stddevval_tf <- semi_structured_tf %>% stddev(first_obs) %>% c
stddevval_torch <- semi_structured_torch %>% stddev(first_obs) %>% c

q05_tf <- semi_structured_tf %>% quant(data = first_obs, probs = 0.05) %>% c
q95_tf <- semi_structured_tf %>% quant(data = first_obs, probs = 0.95) %>% c

q05_torch <- semi_structured_torch %>% quant(data = first_obs, probs = 0.05) %>% c
q95_torch <- semi_structured_torch %>% quant(data = first_obs, probs = 0.95) %>% c

xseq <- seq(q05_tf-1, q95_tf+1, l=1000)
plot(xseq, sapply(xseq, function(x) c(
  as.matrix(dist1_tf$tensor_distribution$prob(x)))),
  type="l", ylab = "density(price)", xlab = "price")
abline(v = c(q05_tf, meanval_tf, q95_tf), col="red", lty=2)

xseq <- seq(q05_torch-1, q95_torch+1, l=1000)
plot(xseq, sapply(xseq, function(x) c(
  as.matrix(exp(dist1_torch$log_prob(x))))),
  type="l", ylab = "density(price)", xlab = "price")
abline(v = c(q05_torch, meanval_torch, q95_torch), col="red", lty=2)

cbind("tf" = semi_structured_tf %>% log_score() %>% as.vector(),
      "torch"= semi_structured_torch %>% log_score())








