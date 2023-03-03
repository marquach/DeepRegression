library(deepregression)
library(torch)
library(luz)
library(mgcv) # just used to fit gam model
library(ggplot2)
library(gamlss)
devtools::load_all("deepregression-main/")
source('Scripts/deepregression_functions.R')

# Up to this moment (02.27.2023), the torch implementation is only able to handle
# deepregression models which are build without the orthogonalizion.
orthog_options = orthog_control(orthogonalize = F)

# Fit some simple models like linear, additive and deep models with
# gam, deepregression tf and deepregression torch

# First a toy example with known true coefficients
x <- seq(from = 0, to = 1, 0.001)
beta1 <- 2
set.seed(42)
y <- rnorm(n = 1001, mean = 0*x, sd = exp(beta1*x))
plot(x,y)

toy_data <- data.frame(x = x, y = y)

intercept_only_torch <-  deepregression(
  list_of_formulas = list(loc = ~ 1, scale = ~ 1),
  data = toy_data, y = toy_data$y,
  orthog_options = orthog_options, return_prepoc = F, 
  subnetwork_builder = subnetwork_init_torch,
  model_builder = torch_dr,
  engine = "torch")

attributes(intercept_only_torch$model)$class #is a luz module generator 
# so it can handle all the stuff from luz

intercept_only_torch$model <- intercept_only_torch$model  %>%
  set_opt_hparams(lr = 0.1)
intercept_only_torch %>% fit(epochs = 500, early_stopping = T)

intercept_only_gamlss <- gamlss(formula = y ~ 1,
                                sigma.formula = ~ 1, family = NO)
cbind(
  'gamlss' = c(intercept_only_gamlss$mu.coefficients,
               intercept_only_gamlss$sigma.coefficients),
  'torch' = lapply(intercept_only_torch$model()[[1]]$parameters, as.array))

# Now with linear term but without intercept
toy_gamlss <- gamlss(formula = y ~ -1+x,
       sigma.formula = ~ -1 + x, family = NO)

mod_torch <- deepregression(
  list_of_formulas = list(loc = ~ -1 + x, scale = ~ -1 + x),
  data = toy_data, y = toy_data$y, orthog_options = orthog_options, return_prepoc = F, 
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  engine = "torch")

mod_torch
# default lr = 0.001 change to 0.1 
mod_torch$model <- mod_torch$model  %>% set_opt_hparams(lr = 0.1)
# torch approach does have a generic fit function 
mod_torch %>% fit(epochs = 500, early_stopping = T)

summary(toy_gamlss)

cbind(
  'gamlss' = c(toy_gamlss$mu.coefficients, toy_gamlss$sigma.coefficients),
  'torch' = lapply(mod_torch$model()[[1]]$parameters, as.array),
  "true" = c(0,2))
# torch even closer to true parameters


################################################################################
################################################################################
#################### additive models #################### 
#### GAM data mcycles
data(mcycle, package = 'MASS')

# Erst mit mgcv
gam_mgcv <- gam(mcycle$accel ~ 1 + s(times), data = mcycle)
gam_tf <- deepregression(
  list_of_formulas = list(loc = ~ 1 + s(times), scale = ~ 1),
  data = mcycle, y = mcycle$accel, orthog_options = orthog_options, engine = "tf"
  )
gam_torch <- deepregression(
  list_of_formulas = list(loc = ~ 1 + s(times), scale = ~ 1),
  data = mcycle, y = mcycle$accel, orthog_options = orthog_options,
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  engine = "torch")

# adapt learning rate to converge faster
gam_torch$model <- gam_torch$model  %>% set_opt_hparams(lr = 0.1)
gam_tf$model$optimizer$lr <- tf$Variable(0.1, name = "learning_rate")

gam_torch %>% fit(epochs = 4000, early_stopping = T)
gam_tf %>% fit(epochs = 4000, early_stopping = T)
# Interesting (does not converge when validation_loss is used as early_stopping)

mean((mcycle$accel - fitted(gam_mgcv) )^2)
# loss is not MSE so loss of model not (directly) comparable
# Loss in deepregression is neg.loglikehood of given distribution
mean((mcycle$accel - gam_tf %>% fitted())^2)

plot(mcycle$times, mcycle$accel)
lines(mcycle$times, fitted(gam_mgcv), col = "red")
lines(mcycle$times, gam_tf %>% fitted(), col = "green")

# there is no generic fitted function for torch
# so we have to extract the data_trafo and transform them to tensors and 
# afterwards save them in a list and push them through the forward function
gam_data <- gam_torch$init_params$parsed_formulas_contents$loc[[1]]$data_trafo()
int_data <- gam_torch$init_params$parsed_formulas_contents$loc[[2]]$data_trafo()
gam_data <- torch_tensor(gam_data)
int_data <- torch_tensor(int_data)
mu_inputs_list <- list(gam_data, int_data)

mean( 
  (mcycle$accel - as.array(gam_torch$model()[[1]][[1]]$forward(mu_inputs_list)))^2)
lines(mcycle$times, gam_torch$model()[[1]][[1]]$forward(mu_inputs_list),
      col = "blue")

gam_tf %>% fit(epochs = 2000, early_stopping = F, validation_split = 0)

plot(mcycle$times, mcycle$accel)
lines(mcycle$times, gam_mgcv %>% fitted(), col = "red")
lines(mcycle$times, gam_tf %>% fitted(), col = "green")
lines(mcycle$times, gam_torch$model()[[1]][[1]]$forward(mu_inputs_list),
      col = "blue")

cbind( 
  "gam" = c(coef(gam_mgcv)[2:10],coef(gam_mgcv)[1]) ,
  "tf" = unlist(gam_tf %>% coef()),
  "torch" = unlist(lapply(gam_torch$model()$parameters[1:2],
                          function(x) t(as.array(x)))))

# showcase from the deepregression (intro) presentation
set.seed(42)
n <- 1000
data = data.frame(matrix(rnorm(4*n), c(n,4)))
colnames(data) <- c("x1","x2","x3","xa")

formula <- ~ -1 + deep_model(x1,x2,x3)

nn_tf <- function(x) x %>%
  layer_dense(units = 32, activation = "relu", use_bias = FALSE) %>%
  layer_dense(units = 1, activation = "linear")

nn_torch <- function() nn_sequential(
  nn_linear(in_features = 3, out_features = 32, bias = F),
  nn_relu(),
  nn_linear(in_features = 32, out_features = 1)
)

y <- rnorm(n) + data$xa^2 + data$x1

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

deep_model_tf$model
deep_model_torch

deep_model_tf %>% fit(epochs = 1000, early_stopping = T)
deep_model_torch %>% fit(epochs = 1000, early_stopping = T)


deep_data <- deep_model_torch$init_params$parsed_formulas_contents$loc[[1]]$data_trafo()
deep_data <- torch_tensor(as.matrix(as.data.frame(deep_data)))
mu_inputs_list <- list(deep_data)

plot(deep_model_tf %>% fitted(),
     deep_model_torch$model()[[1]][[1]]$forward(mu_inputs_list),
     xlab = "tensorflow", ylab = "torch")
cor(deep_model_tf %>% fitted(),
    as.array(
      deep_model_torch$model()[[1]][[1]]$forward(mu_inputs_list)))


# structured model (intercept + linear + additve part)

formula <- ~ 1  + s(xa) + x1

structured_tf <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y,orthog_options = orthog_options,
  list_of_deep_models = list(deep_model = nn_tf), engine = "tf"
)

structured_torch <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y, orthog_options = orthog_options,
  list_of_deep_models = list(deep_model = nn_torch),
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  engine = "torch"
)

# Summary
structured_tf$model
structured_torch

structured_torch$model <- structured_torch$model  %>% set_opt_hparams(lr = 0.1)
structured_tf$model$optimizer$lr <- tf$Variable(0.1, name = "learning_rate")

structured_tf %>% fit(epochs = 1000, early_stopping = T, batch_size = 32)
structured_torch %>% fit(epochs = 1000, early_stopping = T, batch_size = 32)
structured_gam <- gam(y ~ 1 + s(xa) + x1, data = data)

mean((y - structured_gam %>% fitted())^2)

gam_data <- structured_torch$init_params$parsed_formulas_contents$loc[[1]]$data_trafo()
lin_data <- structured_torch$init_params$parsed_formulas_contents$loc[[2]]$data_trafo()
int_data <- structured_torch$init_params$parsed_formulas_contents$loc[[3]]$data_trafo()
lin_data <- torch_tensor(lin_data)
gam_data <- torch_tensor(gam_data)
int_data <- torch_tensor(int_data)

mu_inputs_list <- list(gam_data,
                       lin_data,
                       int_data)

mean((y - structured_tf %>% fitted())^2)
mean((y - as.array(structured_torch$model()[[1]][[1]]$forward(mu_inputs_list)))^2)

fitted_gam <- structured_gam %>% fitted()
fitted_torch <- as.array(structured_torch$model()[[1]][[1]]$forward(mu_inputs_list))
fitted_tf <- structured_tf %>% fitted()

fitted_structured <- data.frame("gamlss" = fitted_gam,
                                "torch" = fitted_torch,
                                "tf" = fitted_tf)
round(cor(fitted_structured),3)

plot(structured_tf %>% fitted(),
     structured_torch$model()[[1]][[1]]$forward(mu_inputs_list),
     xlab = "tensorflow", ylab = "torch")
plot(fitted_gam,
     fitted_torch,
     xlab = "gamlss", ylab = "torch")

cbind( 
  "gam" = c(coef(structured_gam)[3:11],coef(structured_gam)[1:2]) ,
  "tf" = unlist(structured_tf %>% coef()),
  "torch" = unlist(lapply(structured_torch$model()$parameters[1:3],
                function(x) t(as.array(x)))))

plot(structured_gam)
points(data$xa, as.array(mu_inputs_list[[1]]) %*%
         as.array(structured_tf$model$variables[[1]]), col = "green")
points(data$xa, 
       as.array(structured_torch$model()[[1]][[1]][[1]][[1]]$forward(
         mu_inputs_list[[1]])), col = "red")



# showcase from the deepregression documentation

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
semi_structured_tf$model
semi_structured_torch

semi_structured_tf %>% fit(epochs = 1000, early_stopping = T, batch_size = 32)
semi_structured_torch %>% fit(epochs = 1000, early_stopping = T, batch_size = 32)

deep_model_data <- semi_structured_torch$init_params$parsed_formulas_contents$loc[[1]]$data_trafo()
gam_data <- semi_structured_torch$init_params$parsed_formulas_contents$loc[[2]]$data_trafo()
lin_data <- semi_structured_torch$init_params$parsed_formulas_contents$loc[[3]]$data_trafo()
int_data <- semi_structured_torch$init_params$parsed_formulas_contents$loc[[4]]$data_trafo()
deep_model_data <- torch_tensor(as.matrix(as.data.frame(deep_model_data)))
lin_data <- torch_tensor(lin_data)
gam_data <- torch_tensor(gam_data)
int_data <- torch_tensor(int_data)

mu_inputs_list <- list(deep_model_data,
                       gam_data,
                       lin_data,
                       int_data)

plot(semi_structured_tf %>% fitted(),
     semi_structured_torch$model()[[1]][[1]]$forward(mu_inputs_list),
     xlab = "tensorflow", ylab = "torch")

cor(semi_structured_tf %>% fitted(),
    as.array(
      semi_structured_torch$model()[[1]][[1]]$forward(mu_inputs_list)))

cbind(
  unlist(semi_structured_tf %>% coef()),
  unlist(lapply(semi_structured_torch$model()$parameters[4:6], 
                function(x) t(as.array(x)))))
