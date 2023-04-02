# Vergleich Speed Tensorflow / Torch Implementierung
library(keras)
library(luz)
library(torch)
library(rbenchmark)
devtools::load_all("deepregression-main/")
source('scripts/deepregression_functions.R')
orthog_options = orthog_control(orthogonalize = F)


#### GAM data mcycles
data(mcycle, package = 'MASS')
plot(mcycle$times, mcycle$accel)
gam_mgcv <- gam(mcycle$accel ~ -1 + s(times), data = mcycle)
gam_data <- model.matrix(gam_mgcv)

model_keras <- keras_model_sequential()
model_keras <- model_keras %>%
  # Adds a densely-connected layer with 64 units to the model:
  layer_dense(units = 1, activation = 'linear', use_bias = F,
              input_shape = 9)

net_torch <- nn_module(
  classname = "torch_example",
  initialize = function(){
   self$fc1 = nn_linear(9, 1, F)
  },
  forward = function(x){
    torch_flatten(self$fc1(x))

    })

model_t <- net_torch()
optimizer <- optim_adam(model_t$parameters)




test_dataset <- dataset(
  name = "torch_example_dataset",
  
  initialize = function() {
    self$data <- self$prepare_test_data()
  },
  
  .getbatch = function(index) {
    
    x <- self$data[index, 2:10]
    y <- self$data[index, 1]
    
    list(x, y)
  },
  
  .length = function() {
    self$data$size()[[1]]
  },
  
  prepare_test_data = function() {
    
    input <- cbind(mcycle$accel, gam_data)
    input <- as.matrix(input)
    torch_tensor(input)
  }
)

test_ds <- test_dataset()
test_dl <- test_ds %>% dataloader(batch_size = 32)

model_keras %>% compile(
  optimizer = 'adam',
  loss = 'mse')
iter <- test_dl$.iter()
b <- iter$.next()

pre_fit <- net_torch %>% luz::setup(loss = nn_mse_loss(),
                               optimizer = optim_adam)

plain_benchmark <- benchmark( 
  "keras" = model_keras %>% 
    fit(gam_data, mcycle$accel, epochs = 1000, batch_size = 32, verbose = F),
  "torch_loop"= 
    for (epoch in 1:1000) {
      l <- c()
      coro::loop(for (b in test_dl) {
        optimizer$zero_grad()
        output <- model_t(b[[1]])
        loss <- nnf_mse_loss(output, b[[2]])
        loss$backward()
        optimizer$step()
        l <- c(l, loss$item())
        })},
  "torch_luz" = pre_fit %>% fit(test_dl, epochs = 1000, verbose = F),
  columns = c("test", "elapsed", "relative"),
  order = "elapsed",
  replications = 2)

deepreg_gam_tf = deepregression(
  list_of_formulas = list(loc = ~ -1 + s(times), scale = ~ 1),
  data = mcycle, y = mcycle$accel, orthog_options = orthog_options,
  engine = "tf")
deepreg_gam_torch <- deepregression(
  list_of_formulas = list(loc = ~ -1 + s(times), scale = ~ 1),
  data = mcycle, y = mcycle$accel, orthog_options = orthog_options,
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  engine = "torch")

deepreg_benchmark <- benchmark(
  "tf" = deepreg_gam_tf %>% fit(epochs = 1000, early_stopping = F,
                                validation_split = 0, verbose = F),
  "torch" = deepreg_gam_torch %>% fit(epochs = 1000, early_stopping = F,
                                validation_split = 0, verbose = F), 
  columns = c("test", "elapsed", "relative"),
  order = "elapsed",
  replications = 2
)
deepreg_benchmark
