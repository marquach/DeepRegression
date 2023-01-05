# Suggested Steps
library(ggplot2)
library(mgcv)
library(luz)
library(torch)
library(deepregression)

# Point 1 (Familiarize yourself with torch) was done in a different script

# Point 2: Build a basic torch model yourself that works modular: #### 

# GAM B-Spline
data(mcycle, package = 'MASS')
ggplot(mcycle, aes(x = times, y = accel)) +
  geom_point() +
  labs(x = "Miliseconds post impact", y = "Acceleration (g)",
       title = "Simulated Motorcycle Accident",
       subtitle = "Measurements of head acceleration")

# Erst mit mgcv
m1 <- gam(accel ~ -1 + s(times), data = mcycle, family = gaussian)
gam_splines <- model.matrix(m1) # Hier ein bisschen getrickst
dim(gam_splines)

# Jetzt mit TorchR
library(torch)

net <- nn_module(
  "spline_nn",
  initialize = function() {
    self$fc1 <- nn_linear(in_features = 9, #dim(gam_splines)[2]
                          out_features = 1, bias = F)
  },
  
  forward = function(x) {
    x %>%  self$fc1()
  }
)

spline_approach <- net()
# and also need an optimizer
# for adam, need to choose a much higher learning rate in this problem
optimizer <- optim_adam(params = spline_approach$parameters, lr = 0.2)

# network parameters 
spline_approach$parameters # random 
times_torch <- torch_tensor(gam_splines)
accel_torch <- torch_tensor(mcycle$accel)
### training loop 
for (t in 1:1000) {
  ### -------- Forward pass -------- 
  y_pred <- spline_approach(times_torch)$view(133)
  ### -------- compute loss -------- 
  loss <- nnf_mse_loss(y_pred,
                       accel_torch)  #+0.01*spline_approach$parameters$fc1.weight$abs()$sum()
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
cbind(t(as.array(spline_approach$parameters$fc1.weight)),
coef(m1))

loss
mean((m1$fitted.values - mcycle$accel)^2)

plot(mcycle$times, mcycle$accel)
lines(mcycle$times, spline_approach(gam_splines), col = "green")
lines(mcycle$times, m1$fitted.values, col = "red")

# Point 3: Create a spline layer such as the one given in deepregression and ####
# try to replicate the output of a GAM model.

## Folgende Funktion soll umgewandelt werden von TF zu Torch
#def layer_spline(P, units, name, trainable = TRUE,
#                 kernel_initializer = "glorot_uniform"):
#  
#  return(tf.keras.layers.Dense(units = units, name = name, use_bias=False,
#                               kernel_regularizer = squaredPenalty(P, 1),
#                               trainable = trainable,
#                               kernel_initializer = kernel_initializer))

# tf:layers.Dense <==> torch:nn_linear
# Direkt bei nn_linear kein kernel_initializer möglich (nur indirekt: nn_init_)

# kernel_regularizer = squaredPenalty(P, 1) gibt es nicht bei Torch
# Regularisierung direkt bei Loss Berechnung??

source(file = "Scripts/deepregression_functions.R")

test_model <- layer_spline_torch(units = 9, name = "spline_layer")
optimizer <- optim_adam(params = test_model$parameters, lr = 0.2)

### training loop
for (t in 1:1000) {
  ### -------- Forward pass -------- 
  y_pred <- test_model(gam_splines)
  ### -------- compute loss -------- 
  loss <- nnf_mse_loss(y_pred$view(133),
                       accel_torch)  #+0.01*spline_approach$parameters$fc1.weight$abs()$sum()
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
lines(mcycle$times, test_model(gam_splines), col = "purple")


# Jetzt mit luz
luz_mcycle <- nn_module(
  initialize = function() {
    self$spline <- layer_spline_torch(
      units = 9,
      name = "spline_layer")
  },
  forward = function(x) {
      self$spline(x)
  }
)

# ka warum das nicht funktioniert. 
# Sollte ja eigentlich zum gleichen Ergebnis wie oben kommen
# Glaube wegen batchsize bei fit(data=....)
# Lag an Batch_size. Diese ist per Default 32. Hier aber egal da tabellarisch
# Daten. Also alle Daten nutzen, dafür dataloader_options nutzen

test_mcycle <- luz_mcycle %>% 
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam) %>% 
  set_opt_hparams(lr = 0.2) %>%
  fit(
    data = list(
      gam_splines,
      accel_torch$view(c(133, 1))
      ),
    epochs = 1000,
    dataloader_options = list(batch_size = 133)
  )

lines(mcycle$times, predict(test_mcycle, gam_splines), col = "green")

########### Deep Regression example ########### 

library(deepregression)
set.seed(42)
n <- 1000

data = data.frame(matrix(rnorm(4*n), c(n,4)))
colnames(data) <- c("x1","x2","x3","xa")
y <- rnorm(n) + data$xa^2 + data$x1

formula_test <- ~ -1 + s(xa)
orthog_options = orthog_control(orthogonalize = F)

mod_true_preproc <- deepregression(
  list_of_formulas = list(loc = formula_test, scale = ~ 1),
  data = data, y = y, return_prepoc = T, orthog_options = orthog_options
)

mod_true_preproc$loc[[1]]$input_dim

# Test zum verstehen von data_trafo
sum(
  mod_true_preproc$loc[[1]]$data_trafo() -
    model.matrix(gam( y~ s(xa), family = gaussian, data = data))[,-1]) < 0.00001
# sowas wie cSplineDes 

mod_true_preproc$loc[[1]]$data_trafo()
# Vorher nur  pre-processed data und layers ausgegeben (return_prepoc = T)

mod_true <- deepregression(
  list_of_formulas = list(loc = formula_test, scale = ~ 1),
  data = data, y = y, return_prepoc = F, orthog_options = orthog_options
)

if(!is.null(mod_true)){
  # train for more than 10 epochs to get a better model
  mod_true %>% fit(epochs = 100, early_stopping = TRUE)
  mod_true %>% fitted() %>% head()
  mod_true %>% get_partial_effect(name = "s(xa)")
  mod_true %>% coef()
  mod_true %>% plot()
}

### training with loop --------------------------------------------------------------
test_model <- layer_spline_torch(units = 9, name = "spline_layer")
optimizer <- optim_adam(params = test_model$parameters)
for (t in 1:1000) {
  ### -------- Forward pass -------- 
  y_pred <- test_model(mod_true_preproc$loc[[1]]$data_trafo())
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

points(data$xa, test_model(mod_true_preproc$loc[[1]]$data_trafo()),
       col = "purple")


### Training with luz ####

luz_model <- nn_module(
  initialize = function() {
    self$spline <- layer_spline_torch(
      units = 9, #mod_true_preproc$loc[[1]]$input_dim
      name = "spline_layer", kernel_initializer = "") #glorot_uniform"
  },
  forward = function(x) {
    x %>% 
      self$spline()
  }
  )

luz_test <- luz_model %>% 
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
  ) %>% fit(
    data = list(torch_tensor(mod_true_preproc$loc[[1]]$data_trafo()),
                torch_tensor(y)$view(c(1000,1))),
    epochs = 1000, 
    dataloader_options = list(batch_size = 1000)
  )

mod_true$model$weights

data.frame(
  'test' = t(as.array(luz_test$model$parameters$spline.weight)),
  'true' = as.array(mod_true$model$weights[[1]]))

mod_true %>% plot()
points(data$xa,
       predict(luz_test, mod_true_preproc$loc[[1]]$data_trafo()),
       col = "purple")


# Point 4: Make your code more functional by 
# combining a list of layers as done in subnetwork_init.
load('test_data/Input_subnetwork_init.RData')
str(subnetwork_init_input)
# nicht ausführbar weil tensor fehlt
# debugonce durchklicken und input von subnetwork_init 
# als liste außerhalb speichern (<<-)

# hier wird subnetwork_builder ausgeführt 
# subnetwork_builder ist per default subnetwork_init

#additive_predictors <- lapply(1:length(parsed_formulas_contents), 
#  function(i) subnetwork_builder[[i]](parsed_formulas_contents, 
#      deep_top = orthog_options$deep_top, orthog_fun = orthog_options$orthog_fun, 
#      split_fun = orthog_options$split_fun, 
#     shared_layers = weight_options[[i]]$shared_layers, 
#     param_nr = i, gaminputs = gaminputs))
load('test_data/parsed_formulas_contents.RData')
str(parsed_formulas_contents); length(parsed_formulas_contents)
# subnetwork wird für location und scale gebildet
length(parsed_formulas_contents[[1]]) # 4 Coefs für location der distribution
parsed_formulas_contents[[1]][[1]]
length(parsed_formulas_contents[[1]][[1]])
length(parsed_formulas_contents[[1]][[2]])
length(parsed_formulas_contents[[1]][[3]])
length(parsed_formulas_contents[[1]][[4]])
# Je nach Termn unterschiedlich lang

parsed_formulas_contents[[1]][[1]]$data_trafo() %>% str() # Input von NN gleich
parsed_formulas_contents[[1]][[2]]$data_trafo() %>% dim() # Input von s(xa) mit Basis
parsed_formulas_contents[[1]][[3]]$data_trafo() %>% dim()
# Natürlich werden nur Inputdaten bei bei Spline angepasst, wegen Basis Funktionen

length(parsed_formulas_contents[[2]]) # 1 Coefs für scale

# subnetwork_init baut also für jeden Verteilungsparameter ein (Sub)-Netzwerk

debugonce(subnetwork_init)
do.call(subnetwork_init, input_subnet[[1]])

str(input_subnet[[1]])



# Hier mal ein paar Tests um zu verstehen wie subnetzwerke geadded werden
# Try to fit intercept
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
    self$spline <- layer_spline_torch(units = 9, #mod_true_preproc$loc[[1]]$input_dim
                                      name = "spline_layer")
  #self$weightx2 <- nn_linear(in_features = 1, out_features = 1, bias = F)
  #self$weightx1 <- nn_linear(in_features = 1, out_features = 1, bias = F)
  #self$intercept <- nn_parameter(torch_zeros(1)) # Bias
                          },
  forward = function(x) {
    self$spline(x)}
  )

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

ensemble_module <- nn_module(
  
  classname = "ensemble_module",
  
  initialize = function(module1, module2, module3) {
    self$module1 = module1
    self$module2 = module2
    self$module3 = module3
    
  },
  forward = function(data_module_x1, data_module_x2, data_module_x3) {
      self$module1(data_module_x1) +
      self$module2(data_module_x2) +
      self$module3(data_module_x3)
  }
)

test_ensemble <- ensemble_module(spline_layer(), intercept_layer(), x1_layer())
test_ensemble
optimizer <- optim_adam(params = test_ensemble$parameters)
for (t in 1:1000) {
  ### -------- Forward pass -------- 
  y_pred <- 
  test_ensemble(
    data_module_x1 = torch_tensor(mod_true_preproc$loc[[1]]$data_trafo()),
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




