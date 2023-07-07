# Showcases Presentation

library(deepregression)
library(torch)
library(luz)
library(mgcv) # just used to fit gam model
library(ggplot2)
library(gamlss)
library(rbenchmark)
devtools::load_all("deepregression-main/")
source('scripts/deepregression_functions.R')
orthog_options = orthog_control(orthogonalize = F)

# First a toy example with known true coefficients
x <- seq(from = 0, to = 1, 0.001)
beta1 <- 2
set.seed(42)
y <- rnorm(n = 1001, mean = 0*x, sd = exp(beta1*x))
toy_data <- data.frame(x = x, y = y)
plot(toy_data$x, toy_data$y)

toy_gamlss <- gamlss(formula = y ~ -1+x,
                     sigma.formula = ~ -1 + x, family = NO)

mod_torch <- deepregression(
  list_of_formulas = list(loc = ~ -1 + x, scale = ~ -1 + x),
  data = toy_data, y = toy_data$y, orthog_options = orthog_options,
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  engine = "torch")
mod_tf <- deepregression(
  list_of_formulas = list(loc = ~ -1 + x, scale = ~ -1 + x),
  data = toy_data, y = toy_data$y, orthog_options = orthog_options,
  engine = "tf")

mod_torch
mod_tf
# default lr = 0.001 change to 0.1 
mod_torch$model <- mod_torch$model  %>% set_opt_hparams(lr = 0.1)
mod_tf$model$optimizer$lr <- tf$Variable(0.1, name = "learning_rate")

# torch approach does have a generic fit function 
mod_torch %>% fit(epochs = 50, early_stopping = T)
mod_tf %>% fit(epochs = 50, early_stopping = T)

cbind(
  'gamlss' = c(toy_gamlss$mu.coefficients, toy_gamlss$sigma.coefficients),
  'torch' = c(unlist(mod_torch %>% coef(which_param = 1)),
              unlist(mod_torch %>% coef(which_param = 2))),
  'tf' = c(unlist(mod_tf %>% coef(which_param = 1)),
              unlist(mod_tf %>% coef(which_param = 2))),
  "true" = c(0,2))
# torch even closer to true parameters

#### GAM data mcycles
data(mcycle, package = 'MASS')
plot(mcycle$times, mcycle$accel)
gam_mgcv <- gam(mcycle$accel ~ 1 + s(times), data = mcycle)

# Erst mit mgcv

# model build
model_build_benchmark <- benchmark(
  "tf" = gam_tf <- deepregression(
  list_of_formulas = list(loc = ~ 1 + s(times), scale = ~ 1),
  data = mcycle, y = mcycle$accel, orthog_options = orthog_options,
  engine = "tf"),
  "torch"= gam_torch <- deepregression(
  list_of_formulas = list(loc = ~ 1 + s(times), scale = ~ 1),
  data = mcycle, y = mcycle$accel, orthog_options = orthog_options,
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  engine = "torch"),
  columns = c("test", "elapsed", "relative"), replications = 1,
  order = "elapsed")

gam_tf
gam_torch

# adapt learning rate to converge faster
gam_torch$model <- gam_torch$model  %>% set_opt_hparams(lr = 0.1)
gam_tf$model$optimizer$lr <- tf$Variable(0.1, name = "learning_rate")

model_fit_benchmark <- benchmark(
  "torch" = gam_torch %>% fit(epochs = 200, early_stopping = F,
                              validation_split = 0, verbose = F),
  "tf" = gam_tf %>% fit(epochs = 200, early_stopping = F, validation_split = 0, 
                 verbose = F),
  columns = c("test", "elapsed", "relative"), replications = 1,
  order = "elapsed")
  
# Interestly tf does not converge when validation_loss is used as early_stopping,
# because it stop way to early (like after 30 epochs)


mean((mcycle$accel - fitted(gam_mgcv) )^2)
# loss is not MSE so loss of model not (directly) comparable
# Loss in deepregression is neg.loglikehood of given distribution
mean((mcycle$accel - gam_tf %>% fitted())^2)
mean((mcycle$accel - gam_torch %>% fitted())^2)


plot(mcycle$times, mcycle$accel)
lines(mcycle$times, fitted(gam_mgcv), col = "red")
lines(mcycle$times, gam_tf %>% fitted(), col = "green")
lines(mcycle$times, gam_torch %>% fitted(),
      col = "blue")

cbind( 
  "gam" = c(coef(gam_mgcv)[2:10],coef(gam_mgcv)[1]) ,
  "tf" = unlist(gam_tf %>% coef()),
  "torch" = unlist(gam_torch %>% coef()))


plot(gam_torch)
plot(gam_tf)


# showcase from the deepregression (intro) presentation
set.seed(42)
n <- 1000
data = data.frame(matrix(rnorm(4*n), c(n,4)))
colnames(data) <- c("x1","x2","x3","xa")
y <- rnorm(n) + data$xa^2 + data$x1

formula_deep <- ~ -1 + deep_model(x1,x2,x3)

nn_tf <- function(x) x %>%
  layer_dense(units = 32, activation = "relu", use_bias = FALSE) %>%
  layer_dense(units = 1, activation = "linear")

nn_torch <- function() nn_sequential(
  nn_linear(in_features = 3, out_features = 32, bias = F),
  nn_relu(),
  nn_linear(in_features = 32, out_features = 1)
)

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

# Summary
semi_structured_tf
semi_structured_torch


model_fit_benchmark <- benchmark(
  "torch" = semi_structured_torch %>% fit(epochs = 100, early_stopping = F,
                                          verbose = F ),
  "tf" = semi_structured_tf %>% fit(epochs = 100, early_stopping = F,
                                    verbose = F), 
  columns = c("test", "elapsed", "relative"), replications = 1,
  order = "elapsed")


plot(semi_structured_tf %>% fitted(),
     semi_structured_torch %>% fitted(),
     xlab = "tensorflow", ylab = "torch")
cor(semi_structured_tf %>% fitted(),
    semi_structured_torch %>% fitted())

semi_structured_tf %>% coef()
semi_structured_torch %>% coef()

