# simplyconnected_layer_torch Verstehen
library(torch)
source("scripts/deepregression_functions.R")
# Tib linear lasso

# benutz Kombitation aus Layern
# simply connected mit l2 loss 
# und dense layer ebenfalls l2 loss

simplyconnected_layer_torch <- function(la = la,
                                        multfac_initializer = torch_ones,
                                        input_shape){
  nn_module(
  classname = "simply_con",
  
  initialize = function(){
            self$la <- torch_tensor(la)
            self$multfac_initializer <- multfac_initializer
            
            sc <- nn_parameter(x = self$multfac_initializer(input_shape))
            
            sc$register_hook(function(grad){
              grad + la * 2 *sc$sum()
            })
            self$sc <- sc
          },
  
  forward = function(dataset_list)
    torch_multiply(self$sc, dataset_list)$view(
      c(length(dataset_list), length(self$sc))))
}


tiblinlasso_layer_torch <- function(la, input_shape = 1, units = 1){
  
  la <- torch_tensor(la)
  tiblinlasso_layer <- nn_linear(in_features = input_shape,
                                 out_features = units, bias = F)
  
  tiblinlasso_layer$parameters$weight$register_hook(function(grad){
    grad + la*2*tiblinlasso_layer$parameters$weight$sum()
  })
  tiblinlasso_layer
}

tib_layer_torch <-
  nn_module(
  classname = "TibLinearLasso_torch",
  initialize = function(units, la, input_shape, multfac_initializer = torch_ones){
    self$units = units
    self$la = la
    self$multfac_initializer = multfac_initializer
    
    self$fc <- tiblinlasso_layer_torch(la = self$la,
                                       input_shape = input_shape,
                                       units = self$units)
    self$sc <- simplyconnected_layer_torch(la = self$la,
                                            input_shape = input_shape,
                                            multfac_initializer = 
                                              self$multfac_initializer)()
    
  },
  forward = function(data_list){
    self$fc(self$sc(data_list))
  }
)


tibx1 <- tib_layer_torch(units = 1, la = 0.1, input_shape = 1)
tibx1$debug("initialize")
tibx1()
tib_layer_torch()()
#simply_connected_layer_torch <- function(la, multfac_initializer = torch_ones, 
#                                         input_shape){
#  
#  la <- torch_tensor(la)
#  sc_layer <- nn_parameter(x = multfac_initializer(input_shape))
#  
#  sc_layer$register_hook(function(grad){
#    grad + la * 2 *sc_layer$sum()
#  })
#  sc_layer
#}




#simplyconnected_layer_module <- 
#  nn_module(
#    classname = "simply_connected_module_torch",
#    initialize = function(la, multfac_initializer, input_shape){
#      self$sc <- simply_connected_layer_torch(la = la,multfac_initializer = multfac_initializer,
#                                              input_shape = input_shape)
#    },
#    forward = function(dataset_list)
#      torch_multiply(self$sc, dataset_list)$view(
#        c(length(dataset_list), length(self$sc)))
#  )



tib_x2 <- TibLinearLasso_torch(units = 1, la = 0.1, input_shape = 1)

tib_x1x2 <- TibLinearLasso_torch(units = 1, la = 0.1, input_shape = 2); tib_x1x2

tib_x3 <- TibLinearLasso_torch(units = 1, la = 0.1, input_shape = 1)
tib_x4 <- TibLinearLasso_torch(units = 1, la = 0.1, input_shape = 1)
tib_x5 <- TibLinearLasso_torch(units = 1, la = 0.1, input_shape = 1)


set.seed(42)
n <- 1000
p <- 5
x <- data.frame(runif(n*p) %>% matrix(ncol=p))
y <- 10*x[,1] + rnorm(n)

submodules <- list(tib_x1, tib_x2)
submodules <- list(tib_x1x2)
test_model <- model_torch(submodules_list = submodules)

x_input <- as.list(x[,1:2])
test_model$debug("forward")

test_model()$forward(x_input)

optimizer <- optim_adam(params = test_model()$parameters)
for (t in 1:2000) {
  ### -------- Forward pass -------- 
  y_pred <- 
    test_model()(x_input)
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


round(as.array(Reduce(f = "prod", test_model()$parameters[1:2])))
round(as.array(Reduce(f = "prod", test_model()$parameters[3:4])))
round(as.array(Reduce(f = "prod", test_model()$parameters[5:6])))
round(as.array(Reduce(f = "prod", test_model()$parameters[7:8])))
round(as.array(Reduce(f = "prod", test_model()$parameters[9:10])))


debugonce(deepregression)
colnames(x) <- paste("x", 1:5, sep = "")
lasso_test <- deepregression(y = matrix(y), list_of_formulas = list(
  loc = ~ -1 + lasso(x1) + lasso(x2) + lasso(x3) + lasso(x4) + lasso(x5),
  scale = ~ 1), data = x, engine = "torch",
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  orthog_options = orthog_control(orthogonalize = F)
)

lasso_test
debugonce(fit)
lasso_test %>% fit(epochs = 500, early_stopping = F)
round(as.array(Reduce(f = "prod", lasso_test$model()$parameters[1:2])))
round(as.array(Reduce(f = "prod", lasso_test$model()$parameters[3:4])))
round(as.array(Reduce(f = "prod", lasso_test$model()$parameters[5:6])))
round(as.array(Reduce(f = "prod", lasso_test$model()$parameters[7:8])))
round(as.array(Reduce(f = "prod", lasso_test$model()$parameters[9:10])))











