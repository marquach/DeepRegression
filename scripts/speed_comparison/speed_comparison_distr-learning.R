library(torch)
library(luz)
library(rbenchmark)
devtools::load_all("deepregression-main/")
source('scripts/deepregression_functions.R')
orthog_options = orthog_control(orthogonalize = F)

set.seed(42)
n <- 100
x <- runif(n)
y <- rnorm(n = n, mean = 2*x, sd = 1)  + 1
data <- data.frame(x = x)
formula <- ~ 1 + x # orthogonalization has to used in this example

semi_structured_torch <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y, orthog_options = orthog_options,
  engine = "torch")

semi_structured_tf <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y, orthog_options = orthog_options,
  engine = "tf")

# model for loop
data_plain <- data
data_plain <- cbind(1, data_plain)

plain_torch_loop <- nn_module(
  initialize = function(){
    self$linear <- nn_linear(in_features = 1, out_features = 1, T)
    self$intercept_scale <- nn_linear(1, 1, F)
  },
  
  forward = function(input){
    loc <- self$linear(input[,2, drop = F])
    scale <- self$intercept_scale(input[,1, drop = F])
    distr_normal(loc = loc, scale = torch_exp(scale))
  }
)

model <- plain_torch_loop()

# setup for luz
# same as in deepregression (initialized outside)
intercept_loc <- layer_dense_torch(input_shape = 1, units = 1)
linear <- layer_dense_torch(input_shape = 1, units = 1)
intercept_scale <- layer_dense_torch(input_shape = 1, units = 1)

plain_torch_luz <- nn_module(
  initialize = function(){
    self$intercept_loc <- intercept_loc
    self$linear <- linear
    self$intercept_scale <- intercept_scale
  },
  forward = function(input){
    loc <-  self$intercept_loc(input[,1, drop = F] ) + self$linear(input[,2, drop = F])
    scale <- torch_add(1e-8, torch_exp(self$intercept_scale(input[,1, drop = F])))
    
    distr_normal(loc = loc, scale = scale)
  }
)

plain_luz <- luz::setup(plain_torch_luz, 
                        loss = function(input, target){
                                 torch_mean(-input$log_prob(target))},
                        optimizer = optim_adam)

# adapt learning rate to same as loop
plain_luz <- plain_luz %>% set_opt_hparams(lr = 0.1)
semi_structured_torch$model <- semi_structured_torch$model  %>%
  set_opt_hparams(lr = 0.1)
semi_structured_tf$model$optimizer$lr <- 
  tf$Variable(0.1, name = "learning_rate")

epochs <- 1000
plain_deepreg_semistruc_benchmark_n100 <- benchmark(
  "torch_plain" = plain_loop_fit_function(model = model, epochs = epochs, 
                                          batch_size = 32,
                                          data_x = data_plain, data_y = y,
                                          validation_split = 0),
  "plain_torch_luz" = plain_luz_fitted <- fit(plain_luz,
                                              data = list(as.matrix(data_plain), 
                                                          as.matrix(y)),
                                              epochs = epochs, verbose = F,
                                              dataloader_options = 
                                                list(batch_size = 32)),
  "deepregression_luz" = semi_structured_torch %>% fit(epochs = epochs,
                                                       early_stopping = F,
                                                       validation_split = 0,
                                                       verbose = F, batch_size = 32), 
  "deepregression_tf" = semi_structured_tf %>% fit(epochs = epochs, 
                                                   early_stopping = F,
                                                   validation_split = 0, verbose = F,
                                                   batch_size = 32),
  columns = c("test", "elapsed", "relative"),
  order = "elapsed",
  replications = 1
)

plain_deepreg_semistruc_benchmark_n100
results_n100 <- cbind("plain_loop" = rev(lapply(model$parameters, as.array)[1:2]),
      "plain_luz" = lapply(plain_luz_fitted$model$parameters, as.array)[1:2],
      "deepreg_torch" = rev(coef(semi_structured_torch,1)),
      "deepreg_tf" = rev(coef(semi_structured_tf,1)),
      "gamlss" =coef(gamlss::gamlss(y~x, data = data)))


set.seed(42)
n <- 1000
x <- runif(n)
y <- rnorm(n = n, mean = 2*x, sd = 1)  + 1
data <- data.frame(x = x)

semi_structured_torch <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y, orthog_options = orthog_options,
  engine = "torch")

semi_structured_tf <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y, orthog_options = orthog_options,
  engine = "tf")

# model for loop
data_plain <- data
data_plain <- cbind(1, data_plain)

plain_torch_loop <- nn_module(
  initialize = function(){
    self$linear <- nn_linear(in_features = 1, out_features = 1, T)
    self$intercept_scale <- nn_linear(1, 1, F)
  },
  
  forward = function(input){
    loc <- self$linear(input[,2, drop = F])
    scale <- self$intercept_scale(input[,1, drop = F])
    distr_normal(loc = loc, scale = torch_exp(scale))
  }
)

model <- plain_torch_loop()

# setup for luz
# same as in deepregression (initialized outside)
intercept_loc <- layer_dense_torch(input_shape = 1, units = 1)
linear <- layer_dense_torch(input_shape = 1, units = 1)
intercept_scale <- layer_dense_torch(input_shape = 1, units = 1)

plain_torch_luz <- nn_module(
  initialize = function(){
    self$intercept_loc <- intercept_loc
    self$linear <- linear
    self$intercept_scale <- intercept_scale
  },
  forward = function(input){
    loc <-  self$intercept_loc(input[,1, drop = F] ) + self$linear(input[,2, drop = F])
    scale <- torch_add(1e-8, torch_exp(self$intercept_scale(input[,1, drop = F])))
    
    distr_normal(loc = loc, scale = scale)
  }
)


plain_luz <- luz::setup(plain_torch_luz, 
                        loss = function(input, target){
                          torch_mean(-input$log_prob(target))},
                        optimizer = optim_adam)

# adapt learning rate to same as loop
plain_luz <- plain_luz %>% set_opt_hparams(lr = 0.1)
semi_structured_torch$model <- semi_structured_torch$model  %>%
  set_opt_hparams(lr = 0.1)
semi_structured_tf$model$optimizer$lr <- 
  tf$Variable(0.1, name = "learning_rate")

epochs <- 1000
plain_deepreg_semistruc_benchmark_n1000 <- benchmark(
  "torch_plain" = plain_loop_fit_function(model = model, epochs = epochs, 
                                          batch_size = 32,
                                          data_x = data_plain, data_y = y,
                                          validation_split = 0),
  "plain_torch_luz" = plain_luz_fitted <- fit(plain_luz,
                                              data = list(as.matrix(data_plain), 
                                                          as.matrix(y)),
                                              epochs = epochs, verbose = F,
                                              dataloader_options = 
                                                list(batch_size = 32)),
  "deepregression_luz" = semi_structured_torch %>% fit(epochs = epochs,
                                                       early_stopping = F,
                                                       validation_split = 0,
                                                       verbose = F,
                                                       batch_size = 32), 
  "deepregression_tf" = semi_structured_tf %>% fit(epochs = epochs, 
                                                   early_stopping = F,
                                                   validation_split = 0, verbose = F,
                                                   batch_size = 32),
  columns = c("test", "elapsed", "relative"),
  order = "elapsed",
  replications = 1
)

results_n1000 <- cbind("plain_loop" = rev(lapply(model$parameters, as.array)[1:2]),
      "plain_luz" = lapply(plain_luz_fitted$model$parameters, as.array)[1:2],
      "deepreg_torch" = rev(coef(semi_structured_torch,1)),
      "deepreg_tf" = rev(coef(semi_structured_tf,1)),
      "gamlss" =coef(gamlss::gamlss(y~x, data = data)))

plain_deepreg_semistruc_benchmark_n100
plain_deepreg_semistruc_benchmark_n1000
plain_deepreg_semistruc_benchmark_n1000[,2]/
  plain_deepreg_semistruc_benchmark_n100[,2]
  
results_n100
results_n1000


