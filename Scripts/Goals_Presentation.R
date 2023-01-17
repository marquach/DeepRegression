
library(ggplot2)
library(mgcv)
library(luz)
library(torch)
library(deepregression)

# Suggested Steps
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

#debugonce(deepregression)
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


formula <- ~ 1 + deep_model(x1,x2,x3) + s(xa) + x1

deep_model <- function(x) x %>%
  layer_dense(units = 32, activation = "relu", use_bias = FALSE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 8, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")


debugonce(deepregression)  
mod <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y,
  list_of_deep_models = list(deep_model = deep_model)
)


load('test_data/Input_subnetwork_init.RData')
str(subnetwork_init_input)
# nicht ausführbar weil tensor fehlt $ s(xa)+TRUE:<pointer: 0x0>
# debugonce durchklicken und input von subnetwork_init 
# als liste außerhalb speichern (<<-)


load('test_data/parsed_formulas_contents.RData')
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

# data_trafo:
extractvar(parsed_formulas_contents[[1]][[1]]$term)
data[extractvar(parsed_formulas_contents[[1]][[1]]$term)]
# zieht sich x1, x2, x3 aus dem Dataframe. (DF Wichtig da kein "," genutzt wird)

# predict_trafo macht das gleiche aber zieht aus dem neuen Daten

# Input von NN gleich
parsed_formulas_contents[[1]][[1]]$data_trafo() %>% str()
#Input von s(xa) mit Basis-Transformation
parsed_formulas_contents[[1]][[2]]$term
parsed_formulas_contents[[1]][[2]]$data_trafo() %>% dim() 

parsed_formulas_contents[[1]][[3]]$term
parsed_formulas_contents[[1]][[3]]$data_trafo() %>% dim()
# Natürlich werden nur Inputdaten bei bei Spline angepasst, 
# wegen Basis Funktionen.

length(parsed_formulas_contents[[2]]) # 1 Coefs für scale

# subnetwork_init baut also für jeden Verteilungsparameter ein (Sub)-Netzwerk
# hierauf parsed_formulas_contents wird subnetwork_builder ausgeführt 
# subnetwork_builder ist per default subnetwork_init

weight_options <- weight_control()[rep(1, length(parsed_formulas_contents))]
subnetwork_builder <- list(subnetwork_init)[rep(1, length(parsed_formulas_contents))]
orthog_options = orthog_control(orthogonalize = F)
# so enthält Penalties für additive Effekte
# - so$defaultSmoothing: deepregression:::defaultSmoothingFun
# Brauchen wir aber nicht 
so <- penalty_control()

so$gamdata <- precalc_gam(lof = list(loc = formula, scale = ~ 1), data = as.list(data),
            controls = so)

# so$gamdata$data_trafos Liste der Länge 1
gaminputs <- makeInputs(so$gamdata$data_trafos, "gam_inp")
# muss verändert werden für Torch
gaminputs_torch <- torch_tensor(so$gamdata$data_trafos$`s(xa)+TRUE`$data_trafo(),
                                dtype = torch_float32())
# oder nur Dimensionen
gaminputs_torch <- torch_empty(parsed_formulas_contents[[1]][[2]]$data_trafo() %>% dim(),
                               dtype = torch_float32())
# gaminputs abgefertigt. Jetzt kommen restlichen unstrukturierte
# und strukturierte Parts, welche nicht gam sind sind

debugonce(subnetwork_builder[[1]])
additive_predictors <- lapply(1:length(parsed_formulas_contents), 
    function(i) subnetwork_builder[[i]](parsed_formulas_contents, 
      deep_top = orthog_options$deep_top, orthog_fun = orthog_options$orthog_fun, 
      split_fun = orthog_options$split_fun, 
      shared_layers = weight_options[[i]]$shared_layers, 
      param_nr = i, gaminputs = gaminputs))


names(additive_predictors) <- names(parsed_formulas_contents)
gaminputs <- list(gaminputs)
names(gaminputs) <- "gaminputs"
additive_predictors <- c(gaminputs, additive_predictors)

# model_builder (aka. keras_dr) setzt subnetworks dann zusammen und reduziert auf eine Dimension
mod

