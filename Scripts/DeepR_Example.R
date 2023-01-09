########### Deep Regression example ########### 

library(deepregression)
set.seed(42)
n <- 1000

data = data.frame(matrix(rnorm(4*n), c(n,4)))
colnames(data) <- c("x1","x2","x3","xa")
y <- rnorm(n) + data$xa^2 + data$x1

# Dürfen (orthogonalize = F) nutzen
orthog_options = orthog_control(orthogonalize = F)

## Easiest Model
formula <- ~ 1
#debugonce(deepregression)  
mod <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y, orthog_options = orthog_options
)
# sollte zwei params haben (loc, scale)
mod
# passt

mod %>% fit(epochs = 100, early_stopping = TRUE)
mod
mean(y)
mod %>% coef()


## Easy Model ohne NN
formula <- ~ 1  + x1
#debugonce(deepregression)  
mod <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y, orthog_options = orthog_options
)
mod

  mod %>% fit(epochs = 100, early_stopping = TRUE)
  mod %>% coef()
  mod %>% plot()

mod
# lustig, dass shape immer noch none bei Input.


formula <- ~ 1 +s(xa) + x1
#debugonce(deepregression)  
mod <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y, orthog_options = orthog_options
)
mod

mod %>% fit(epochs = 100, early_stopping = TRUE)
mod %>% coef()
mod %>% plot()

mod
# Komisch, dass  s_xa__1 (Dense) (None, 1) None anzeigt. 
# Muss ja eigentlich fix (9, 1) sein. Selbes bei den anderen dense layer
# mod output setzt sich immer aus input und 
# layer concatenate und Verteilungsparameter zusammen

formula_test <- ~ -1 + s(xa)

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



##### Verständis was innerhalb von deepregression abläuft ####
# debugonce durchklicken und input von subnetwork_init 
# als liste außerhalb speichern (<<-)

load('test_data/Input_subnetwork_init.RData')
str(subnetwork_init_input)
# nicht ausführbar weil tensor fehlt $ s(xa)+TRUE:<pointer: 0x0>
# wird weiter unten erstellt

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
















### training with loop --------------------------------------------------------------
source("Scripts/deepregression_functions.R")
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