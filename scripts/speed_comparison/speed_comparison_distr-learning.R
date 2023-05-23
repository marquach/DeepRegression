library(torch)
library(luz)
library(rbenchmark)
devtools::load_all("deepregression-main/")
source('scripts/deepregression_functions.R')
orthog_options = orthog_control(orthogonalize = F)

set.seed(42)
n <- 100
data = data.frame(matrix(rnorm(4*n), c(n,4)))
colnames(data) <- c("x1","x2","x3","xa")
y <- rnorm(n) + data$xa^2 + data$x1

formula <- ~ 1 + deep_model(x1,x2,x3) + s(xa) + x1

nn_torch <- nn_module(
  initialize = function(){
    self$fc1 <- nn_linear(in_features = 3, out_features = 32, bias = F)
    self$fc2 <-  nn_linear(in_features = 32, out_features = 1)
  },
  forward = function(x){
    x %>% self$fc1() %>% nnf_relu() %>% self$fc2()
  }
)

semi_structured_torch <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y, orthog_options = orthog_options,
  list_of_deep_models = list(deep_model = nn_torch),
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  engine = "torch")

# model for loop
data_plain <- data
data_plain <- cbind(1, data_plain) #add intercept
data_plain <- cbind(data_plain, #add basis transformation
                    semi_structured_torch$init_params$parsed_formulas_contents$loc[[2]]$data_trafo())

plain_torch_loop <- nn_module(
  initialize = function(){
    
    self$intercept_loc <- layer_dense_torch(input_shape = 1, units = 1)
    self$spline <- layer_spline_torch(P =
                                        semi_structured_torch$init_params$parsed_formulas_contents$loc[[2]]$penalty$values)
    self$nn <- nn_torch()
    self$linear <- layer_dense_torch(input_shape = 1, units = 1)
    self$intercept_scale <- layer_dense_torch(input_shape = 1, units = 1)
  },
  forward = function(x){
    loc <-  self$intercept_loc(x[,1, drop = F] ) +
            self$spline(x[,6:14]) + 
            self$nn(x[,2:4]) +
            self$linear(x[,2, drop = F])
    scale <- torch_add(1e-8, torch_exp(self$intercept_scale(x[,1, drop = F])))
    
    distr_normal(loc = loc, scale =scale)
  }
)

model <- plain_torch_loop()
opt <- optim_adam(model$parameters, lr = 0.1)

intercept_loc <- layer_dense_torch(input_shape = 1, units = 1)
spline <- layer_spline_torch(P =
                     semi_structured_torch$init_params$parsed_formulas_contents$loc[[2]]$penalty$values)
nn <- nn_torch()
linear <- layer_dense_torch(input_shape = 1, units = 1)
intercept_scale <- layer_dense_torch(input_shape = 1, units = 1)

plain_torch_luz <- nn_module(
  initialize = function(){
    
    self$intercept_loc <- intercept_loc
    self$spline <- spline
    self$nn <- nn
    self$linear <- linear
    self$intercept_scale <- intercept_scale
  },
  forward = function(x){
    loc <-  self$intercept_loc(x[,1, drop = F] ) +
            self$spline(x[,6:14]) + 
            self$nn(x[,2:4]) +
            self$linear(x[,2, drop = F])
    scale <- torch_add(1e-8, torch_exp(self$intercept_scale(x[,1, drop = F])))
    
    distr_normal(loc = loc, scale = scale)
  }
)

x <- torch_tensor(as.matrix(data_plain))
plain_luz <- luz::setup(plain_torch_luz, 
                        loss = function(input, target){
                                 torch_mean(-input$log_prob(target))},
                        optimizer = optim_adam)

epochs <- 1000
plain_deepreg_semistruc_benchmark_n100 <- benchmark(
  "torch_plain" = for(i in 1:epochs){
    opt$zero_grad()
    d <- model(x)
    loss <- torch_mean(-d$log_prob(y))
    loss$backward()
    opt$step()},
  "plain_torch_luz" = fit(plain_luz, data = list(x, y), epochs = epochs,
                          verbose = F),
  "deepregression_luz" = semi_structured_torch %>% fit(epochs = epochs,
                                              early_stopping = F,
                                              validation_split = 0,
                                              verbose = F), 
  "deepregression_tf" = semi_structured_tf %>% fit(epochs = epochs, 
                                                  early_stopping = F,
                                    validation_split = 0, verbose = F),
  columns = c("test", "elapsed", "relative"),
  order = "elapsed",
  replications = 1
)

set.seed(42)
n <- 1000
data = data.frame(matrix(rnorm(4*n), c(n,4)))
colnames(data) <- c("x1","x2","x3","xa")
y <- rnorm(n) + data$xa^2 + data$x1

formula <- ~ 1 + deep_model(x1,x2,x3) + s(xa) + x1

nn_torch <- nn_module(
  initialize = function(){
    self$fc1 <- nn_linear(in_features = 3, out_features = 32, bias = F)
    self$fc2 <-  nn_linear(in_features = 32, out_features = 1)
  },
  forward = function(x){
    x %>% self$fc1() %>% nnf_relu() %>% self$fc2()
  }
)

semi_structured_torch <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y, orthog_options = orthog_options,
  list_of_deep_models = list(deep_model = nn_torch),
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  engine = "torch")

# model for loop
data_plain <- data
data_plain <- cbind(1, data_plain) #add intercept
data_plain <- cbind(data_plain, #add basis transformation
                    semi_structured_torch$init_params$parsed_formulas_contents$loc[[2]]$data_trafo())

plain_torch_loop <- nn_module(
  initialize = function(){
    
    self$intercept_loc <- layer_dense_torch(input_shape = 1, units = 1)
    self$spline <- layer_spline_torch(P =
                                        semi_structured_torch$init_params$parsed_formulas_contents$loc[[2]]$penalty$values)
    self$nn <- nn_torch()
    self$linear <- layer_dense_torch(input_shape = 1, units = 1)
    self$intercept_scale <- layer_dense_torch(input_shape = 1, units = 1)
  },
  forward = function(x){
    loc <-  self$intercept_loc(x[,1, drop = F] ) +
      self$spline(x[,6:14]) + 
      self$nn(x[,2:4]) +
      self$linear(x[,2, drop = F])
    scale <- torch_add(1e-8, torch_exp(self$intercept_scale(x[,1, drop = F])))
    
    distr_normal(loc = loc, scale =scale)
  }
)

model <- plain_torch_loop()
opt <- optim_adam(model$parameters, lr = 0.1)

intercept_loc <- layer_dense_torch(input_shape = 1, units = 1)
spline <- layer_spline_torch(P =
                               semi_structured_torch$init_params$parsed_formulas_contents$loc[[2]]$penalty$values)
nn <- nn_torch()
linear <- layer_dense_torch(input_shape = 1, units = 1)
intercept_scale <- layer_dense_torch(input_shape = 1, units = 1)

plain_torch_luz <- nn_module(
  initialize = function(){
    
    self$intercept_loc <- intercept_loc
    self$spline <- spline
    self$nn <- nn
    self$linear <- linear
    self$intercept_scale <- intercept_scale
  },
  forward = function(x){
    loc <-  self$intercept_loc(x[,1, drop = F] ) +
      self$spline(x[,6:14]) + 
      self$nn(x[,2:4]) +
      self$linear(x[,2, drop = F])
    scale <- torch_add(1e-8, torch_exp(self$intercept_scale(x[,1, drop = F])))
    
    distr_normal(loc = loc, scale = scale)
  }
)

x <- torch_tensor(as.matrix(data_plain))
plain_luz <- luz::setup(plain_torch_luz, 
                        loss = function(input, target){
                          torch_mean(-input$log_prob(target))},
                        optimizer = optim_adam)

epochs <- 1000
plain_deepreg_semistruc_benchmark_n1000 <- benchmark(
  "torch_plain" = for(i in 1:10000){
    opt$zero_grad()
    d <- model(x)
    loss <- torch_mean(-d$log_prob(y))
    loss$backward()
    print(loss)
    opt$step()},
  "plain_torch_luz" = fit(plain_luz, data = list(x, y), epochs = epochs,
                          verbose = F),
  "deepregression_luz" = semi_structured_torch %>% fit(epochs = epochs,
                                                       early_stopping = F,
                                                       validation_split = 0,
                                                       verbose = F), 
  "deepregression_tf" = semi_structured_tf %>% fit(epochs = epochs, 
                                                   early_stopping = F,
                                                   validation_split = 0, verbose = F),
  columns = c("test", "elapsed", "relative"),
  order = "elapsed",
  replications = 1
)



