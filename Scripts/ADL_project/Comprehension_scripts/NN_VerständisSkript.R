# Hier mal ein paar Tests um zu verstehen wie subnetzwerke geadded werden
# Try to fit intercept

library(deepregression)
library(torch)
library(luz)
set.seed(42)
n <- 1000

data = data.frame(matrix(rnorm(4*n), c(n,4)))
colnames(data) <- c("x1","x2","x3","xa")
y <- rnorm(n) + data$xa^2 + data$x1

source("Scripts/deepregression_functions.R")
formula_intercept <- ~ 1
orthog_options = orthog_control(orthogonalize = F)

mod_true_w <- deepregression(
  list_of_formulas = list(loc = formula_intercept, scale = ~ 1),
  data = data, y = y, return_prepoc = F, orthog_options = orthog_options
)

if(!is.null(mod_true_w)){
  # train for more than 10 epochs to get a better model
  mod_true_w %>% fit(epochs = 100, early_stopping = TRUE)
  mod_true_w %>% fitted() %>% head()
  mod_true_w %>% coef()
}

luz_model <- nn_module(
  
  initialize = function() {
    #self$spline <- layer_spline_torch(
    # units = 9, #mod_true_preproc$loc[[1]]$input_dim
    # name = "spline_layer")
    self$intercept <- nn_linear(in_features = 1, out_features = 1, bias = F)
  },
  
  forward = function(x) {
    self$intercept(x)
  }
)

luz_test(torch_tensor(rep(1, 1000))$view(c(1000,1)))$view(1000)

luz_test <- luz_model %>% 
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
  ) %>% 
  set_opt_hparams(lr = 0.2) %>%
  fit(
    data = list(
      torch_tensor(rep(1, 1000))$view(c(1000,1)),
      torch_tensor(y)$view(c(1000, 1))
    ),
    epochs = 100, 
    dataloader_options = list(batch_size = 1000)
  )

mod_true_w %>% coef()
luz_test$model$parameters
# Intercept only ist Mittelwert
mean(y)
# Passt also


mod <- deepregression(
  list_of_formulas = list(loc = ~ 1 + x2, scale = ~ 1),
  data = data, y = y, return_prepoc = F, orthog_options = orthog_options
)
if(!is.null(mod)){
  # train for more than 10 epochs to get a better model
  mod %>% fit(epochs = 100, early_stopping = TRUE)
  mod %>% fitted() %>% head()
  mod %>% coef()
}

luz_model <- nn_module(
  
  initialize = function() {
    #self$spline <- layer_spline_torch(
    # units = 9, #mod_true_preproc$loc[[1]]$input_dim
    # name = "spline_layer")
    self$weightx2 <- nn_linear(in_features = 1, out_features = 1, bias = F)
    self$intercept <- nn_parameter(torch_zeros(1)) # Bias
    # Auch ein weg für den Intercept, aber bei uns 1er als input für intercept
  },
  
  forward = function(x) {
    self$weightx2(x) + self$intercept
  }
)


luz_test <- luz_model %>% 
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
  ) %>% 
  set_opt_hparams(lr = 0.2) %>%
  fit(
    data = list(
      torch_tensor(data$x2)$view(c(1000,1)),
      torch_tensor(y)$view(c(1000, 1))
    ),
    epochs = 100, 
    dataloader_options = list(batch_size = 1000)
  )

mod %>% coef()
luz_test$model$parameters
coef(lm(y~1 + x2, data = data))

# Try x1 und x2 einzeln



luz_model <- nn_module(
  
  initialize = function() {
    #self$spline <- layer_spline_torch(
    # units = 9, #mod_true_preproc$loc[[1]]$input_dim
    # name = "spline_layer")
    self$weightx2 <- nn_linear(in_features = 1, out_features = 1, bias = F)
    self$weightx1 <- nn_linear(in_features = 1, out_features = 1, bias = F)
    self$intercept <- nn_parameter(torch_zeros(1)) # Bias
  },
  
  forward = function(x) {
    self$weightx1(x[,1]$view(c(1000,1))) + self$weightx2(x[,2]$view(c(1000,1)))+
      self$intercept
  }
)

luz_test <- luz_model %>% 
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
  ) %>% 
  set_opt_hparams(lr = 0.2) %>%
  fit(
    data = list(
      torch_tensor(as.matrix(data[, 1:2])),
      torch_tensor(y)$view(c(1000, 1))
    ),
    epochs = 100, 
    dataloader_options = list(batch_size = 1000)
  )

coef(lm(y~1 + x1 + x2, data = data))
luz_test$model$parameters

spline_layer <- nn_module(
  classname = "spline_module", 
  
  initialize = function() {
    self$spline <- layer_spline(units = 9, #mod_true_preproc$loc[[1]]$input_dim
                                      name = "spline_layer")
    #self$weightx2 <- nn_linear(in_features = 1, out_features = 1, bias = F)
    #self$weightx1 <- nn_linear(in_features = 1, out_features = 1, bias = F)
    #self$intercept <- nn_parameter(torch_zeros(1)) # Bias
  },
  forward = function(x) {
    self$spline(x)}
)


# Einzelne Layer die unterschiedliche Inputs erhalten
# So sollte passiert es m.M.n. auch bei deepregression
intercept_layer <- nn_module(
  
  classname = "intercept_module",
  
  initialize = function() {
    #self$spline <- layer_spline_torch(units = 9, #mod_true_preproc$loc[[1]]$input_dim
    #                                  name = "spline_layer")
    #self$weightx2 <- nn_linear(in_features = 1, out_features = 1, bias = F)
    #self$weightx1 <- nn_linear(in_features = 1, out_features = 1, bias = F)
    self$intercept <- nn_linear(1, 1, F) # Bias
  },
  forward = function(x) {
    self$intercept(x)
  }
)

x1_layer <- nn_module(
  
  classname = "x1_module",
  
  initialize = function() {
    #self$spline <- layer_spline_torch(units = 9, #mod_true_preproc$loc[[1]]$input_dim
    #                                  name = "spline_layer")
    #self$weightx2 <- nn_linear(in_features = 1, out_features = 1, bias = F)
    self$weightx1 <- nn_linear(in_features = 1, out_features = 1, bias = F)
    #self$intercept <- nn_parameter(torch_zeros(1)) # Bias
  },
  forward = function(x) {
    self$weightx1(x)
  }
)


# unser model_builder

ensemble_module <- nn_module(
  
  classname = "ensemble_module",
  
  initialize = function(module1, module2, module3) {
    self$module1 = module1
    self$module2 = module2
    self$module3 = module3
    
  },
  forward = function(data_module_x1, data_module_x2, data_module_x3) {
    x1 <- self$module1(data_module_x1)
    x2 <- self$module2(data_module_x2) 
    x3 <- self$module3(data_module_x3)
    Reduce(f = "+", list(x1,x2,x3))
  }
)

test_ensemble <- ensemble_module(spline_layer(), intercept_layer(), x1_layer())
optimizer <- optim_adam(params = test_ensemble$parameters)


test_submodules <- list(spline_layer(),
                        intercept_layer(),
                        x1_layer())
self <- nn_module_list(test_submodules)
self$apply(self$initialize)
test[[3]]$forward

#param_nr adden
model_builder <-  nn_module(
     classname = "test_module",
     initialize = function(submodules_list) {
       self$linears <- nn_module_list(submodules_list)
       },
     forward = function(data_list) {
       linears <- lapply(1:length(self$linears), function(x){
         self$linears[[x]](data_list[[x]])
         })
       Reduce(f = "+", x = linears)}
   )

test_builder <- model_builder(list(spline_layer(),
                   intercept_layer(),
                   x1_layer()))

input_list <- list(
  torch_tensor(parsed_formulas_contents$loc[[2]]$data_trafo()),
  torch_tensor(rep(1, 1000))$view(c(1000, 1)),
  torch_tensor(data$x1)$view(c(1000, 1)))
debugonce(test_builder)
test_builder(input_list)

torch_cat(
  c(torch_tensor(parsed_formulas_contents$loc[[2]]$data_trafo()),
    torch_tensor(rep(1, 1000))$view(c(1000, 1)),
    torch_tensor(data$x1)$view(c(1000, 1))))



for (t in 1:1000) {
  ### -------- Forward pass -------- 
  y_pred <- 
    test_ensemble(
      data_module_x1 = torch_tensor(parsed_formulas_contents$loc[[2]]$data_trafo()),
      data_module_x2 = torch_tensor(rep(1, 1000))$view(c(1000, 1)),
      data_module_x3 = torch_tensor(data$x1)$view(c(1000, 1)))
  ### -------- compute loss -------- 
  loss <- nnf_mse_loss(y_pred$view(1000),
                       torch_tensor(y))  #+0.01*spline_approach$parameters$fc1.weight$abs()$sum()
  if (t %% 100 == 0)
    cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")
  
  ### -------- Backpropagation -------- 
  
  # Still need to zero out the gradients before the backward pass, only this time,
  # on the optimizer object
  optimizer$zero_grad()
  
  # gradients are still computed on the loss tensor (no change here)
  loss$backward()
  
  ### -------- Update weights -------- 
  
  # use the optimizer to update model parameters
  optimizer$step()
} 

c(as.array(test_ensemble$parameters$module2.intercept.weight),
  as.array(test_ensemble$parameters$module3.weightx1.weight),
  t(as.array(test_ensemble$parameters$module1.spline.weight)))

gam_model <- (gam(y ~ 1 + x1 + s(xa), data = data, family = gaussian))
gam_model$fitted.values

formula <- ~ 1 + x1 + s(xa)
orthog_options = orthog_control(orthogonalize = F)

mod <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y, return_prepoc = F, orthog_options = orthog_options
)

if(!is.null(mod)){
  # train for more than 10 epochs to get a better model
  mod %>% fit(epochs = 100, early_stopping = TRUE)
  mod %>% fitted() %>% head()
  mod %>% coef()
}

plot(
  as.array(test_ensemble(
    data_module_x1 = torch_tensor(mod_true_preproc$loc[[1]]$data_trafo()),
    data_module_x2 = torch_tensor(rep(1, 1000))$view(c(1000, 1)),
    data_module_x3 = torch_tensor(data$x1)$view(c(1000, 1)))),
  gam_model$fitted.values)

plot(mod %>% fitted(),
     gam_model$fitted.values)

library(luz)
luz::setup
