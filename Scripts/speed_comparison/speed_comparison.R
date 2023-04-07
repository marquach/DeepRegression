# Vergleich Speed Tensorflow / Torch Implementierung
library(keras)
library(luz)
library(torch)
library(rbenchmark)
devtools::load_all("deepregression-main/")
source('scripts/deepregression_functions.R')
orthog_options = orthog_control(orthogonalize = F)

load("scripts/speed_comparison/speed_comparison_dev_version.RData")


#### GAM data mcycles
data(mcycle, package = 'MASS')
plot(mcycle$times, mcycle$accel)
gam_mgcv <- gam(mcycle$accel ~ -1 + s(times), data = mcycle)
gam_data <- model.matrix(gam_mgcv)
# because of nn use_bias first without intercept column
gam_mgcv <- gam(mcycle$accel ~ 1 + s(times), data = mcycle)


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

model_keras <- keras_model_sequential()
model_keras <- model_keras %>%
  # Adds a densely-connected layer with 64 units to the model:
  layer_dense(units = 1, activation = 'linear', use_bias = T,
              input_shape = 9)

net_torch <- nn_module(
  classname = "torch_example",
  initialize = function(){
   self$fc1 = nn_linear(9, 1, T)
  },
  forward = function(x){
    torch_flatten(self$fc1(x))

    })

model_t <- net_torch()
model_t_manual <- net_torch()
optimizer_t <- optim_adam(model_t$parameters)
optimizer_t_manual <- optim_adam(model_t_manual$parameters)

test_ds <- test_dataset()
test_dl <- test_ds %>% dataloader(batch_size = 32)

model_keras %>% compile(
  optimizer = 'adam',
  loss = 'mse')

iter <- test_dl$.iter()
b <- iter$.next()

pre_fit <- net_torch %>% luz::setup(loss = nn_mse_loss(),
                               optimizer = optim_adam)

input <- gam_data
input <- torch_tensor(input)
data_x <- input
data_y <- torch_tensor(mcycle$accel)

batch_size <- 32
num_data_points <- data_y$size(1)
num_batches <- floor(num_data_points/batch_size)

epochs <- 1000

plain_benchmark <- benchmark( 
  "keras" = model_keras %>% 
    fit(gam_data, mcycle$accel, epochs = epochs, batch_size = 32, verbose = F),
  "torch_loop"= 
    for (epoch in 1:epochs) {
      l_loop <- c()
      coro::loop(for (b in test_dl) {
        optimizer_t$zero_grad()
        output <- model_t(b[[1]])
        loss_loop <- nnf_mse_loss(output, b[[2]])
        loss_loop$backward()
        optimizer_t$step()
        l_loop <- c(l_loop, loss_loop$item())
        })
      #cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l_loop)))
      },
  "torch_loop_manual" = 
    for(epoch in 1:epochs){
      l_man <- c()
      # rearrange the data each epoch
      permute <- torch_randperm(num_data_points) + 1L
      data_x <- data_x[permute]
      data_y <- data_y[permute]
      
      # manually loop through the batches
      for(batch_idx in 1:num_batches){
        
        # here index is a vector of the indices in the batch
        index <- (batch_size*(batch_idx-1) + 1):(batch_idx*batch_size)
        
        x <- data_x[index]
        y <- data_y[index]
        
        optimizer_t_manual$zero_grad()
        output <- model_t_manual(x)
        loss_man <- nnf_mse_loss(output, y)
        loss_man$backward()
        optimizer_t_manual$step()
        l_man <- c(l_man, loss_man$item())
      }
      
      #cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l_man)))
    },
  "torch_luz" = pre_fit %>% fit(test_dl, epochs = epochs, verbose = F),
  columns = c("test", "elapsed", "relative"),
  order = "elapsed",
  replications = 2)

model_t_manual$parameters
model_t$parameters
pre_fit()$parameters
model_keras$weights



deepreg_gam_tf = deepregression(
  list_of_formulas = list(loc = ~ 1 + s(times), scale = ~ 1),
  data = mcycle, y = mcycle$accel, orthog_options = orthog_options,
  engine = "tf")
deepreg_gam_torch <- deepregression(
  list_of_formulas = list(loc = ~ 1 + s(times), scale = ~ 1),
  data = mcycle, y = mcycle$accel, orthog_options = orthog_options,
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  engine = "torch")

deepreg_benchmark <- benchmark(
  "tf" = deepreg_gam_tf %>% fit(epochs = epochs, early_stopping = F,
                                validation_split = 0, verbose = F),
  "torch" = deepreg_gam_torch %>% fit(epochs = epochs, early_stopping = F,
                                validation_split = 0, verbose = F), 
  columns = c("test", "elapsed", "relative"),
  order = "elapsed",
  replications = 2
)
deepreg_benchmark

speed_comparison <- list(deepreg_benchmark, plain_benchmark)
save(speed_comparison, file= "scripts/speed_comparison/speed_comparison_dev_version.RData")

