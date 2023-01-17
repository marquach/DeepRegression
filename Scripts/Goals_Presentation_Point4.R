# Goals of presentation
# Point 4

# Point 4: Make your code more functional by 
# combining a list of layers as done in subnetwork_init.

load("test_data/parsed_formulas_contents.RData")

str(parsed_formulas_contents); length(parsed_formulas_contents)
# Damit sollten wir weiterhin arbeiten können

# subnetwork wird für location und scale gebildet
length(parsed_formulas_contents[[1]]) # 4 Coefs für location der distribution
length(parsed_formulas_contents[[1]][[1]])
length(parsed_formulas_contents[[1]][[2]])
length(parsed_formulas_contents[[1]][[3]])
length(parsed_formulas_contents[[1]][[4]])
# Je nach Term (Special) unterschiedlich lang

str(parsed_formulas_contents[[1]][[1]])
# Enthält term, daten, input_dim und layer usw.
parsed_formulas_contents[[1]][[1]]$layer
parsed_formulas_contents[[1]][[2]]$layer

# data_trafo:
extractvar(parsed_formulas_contents[[1]][[1]]$term)
data[extractvar(parsed_formulas_contents[[1]][[1]]$term)]
# zieht sich x1, x2, x3 aus dem Dataframe. (DF Wichtig da kein "," genutzt wird)

# predict_trafo macht das gleiche aber zieht aus dem neuen Datensatz

str(parsed_formulas_contents[[1]])
# Gibt mir also Daten und Input_dim vor.

#
# Versuch mehrere NN zu "schichten"
formula <- y ~ 1 + x1 + s(xa)
gam_model <- mgcv::gam(formula, data = data, family = gaussian)
gam_model$fitted.values


source("Scripts/deepregression_functions.R")
spline_layer <- nn_module(
  classname = "spline_module", 
  
  initialize = function() {
    self$spline <- layer_spline_torch(units = 9, #parsed_formulas_contents[[1]][[2]]$input_dim
                                      name = "spline_layer")
  },
  
  forward = function(x) {
    self$spline(x)}
)

intercept_layer <- nn_module(
  classname = "intercept_module",
  
  initialize = function() {
    self$intercept <- nn_linear(1, 1, F) # Bias
    #self$intercept <- nn_parameter(torch_zeros(1)) So nicht, 
    #da Input von Intercept Layer (1,1,1, ...) ist.
  },
  
  forward = function(x) {
    self$intercept(x)
  }
  
)

x1_layer <- nn_module(
  
  classname = "x1_module",
  
  initialize = function() {
    self$weightx1 <- nn_linear(in_features = 1, out_features = 1, bias = F)
  },
  
  forward = function(x) {
    self$weightx1(x)
  }
)

ensemble_module <- nn_module(
  
  classname = "ensemble_module",
  
  initialize = function(module1, module2, module3) {
    self$module1 = module1
    self$module2 = module2
    self$module3 = module3
    
  },
  forward = function(data_module_1, data_module_2, data_module_3) {
    self$module1(data_module_1) +
      self$module2(data_module_2) +
      self$module3(data_module_3)
  }
)

test_ensemble <- ensemble_module(spline_layer(), intercept_layer(), x1_layer())
test_ensemble

optimizer <- optim_adam(params = test_ensemble$parameters)

sum((model.matrix(gam_model)[,-c(1,2)] - parsed_formulas_contents$loc[[2]]$data_trafo()))
# same

for (t in 1:2500) {
  ### -------- Forward pass -------- 
  y_pred <- 
    test_ensemble(
      data_module_1 = torch_tensor(parsed_formulas_contents$loc[[2]]$data_trafo()), # xa %*% Z
      data_module_2 = torch_tensor(rep(1, 1000))$view(c(1000, 1)), # Intercept
      data_module_3 = torch_tensor(data$x1)$view(c(1000, 1))) # x1
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

loss
mean((gam_model$fitted.values - gam_model$y)^2)
# paar mehr Epochen und ist gleich

c(as.array(test_ensemble$parameters$module2.intercept.weight),
  as.array(test_ensemble$parameters$module3.weightx1.weight),
  t(as.array(test_ensemble$parameters$module1.spline.weight)))
coef(gam_model)

plot(
  as.array(test_ensemble(
    data_module_1 = torch_tensor(parsed_formulas_contents$loc[[2]]$data_trafo()),
    data_module_2 = torch_tensor(rep(1, 1000))$view(c(1000, 1)),
    data_module_3 = torch_tensor(data$x1)$view(c(1000, 1)))),
  gam_model$fitted.values)


# Coefs unterschiedlich aber prediction ähnlich ?!


# Wichtig
layer_generator
gam_processor



