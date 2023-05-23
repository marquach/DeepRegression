#grouped lasso test
library(torch)
library(luz)
library(caret)
library(gglasso)
devtools::load_all("deepregression-main/")
source("scripts/deepregression_functions.R")

set.seed(42)
tf$random$set_seed(42)
n <- 1000
x <- sample(1:10, size = n, replace = T)
x <- as.factor(x)
x <- data.frame(x)

x_help <- dummyVars(" ~ .", data = x)
x_onehot <- data.frame(predict(x_help, newdata = x))
beta <- c(0,10,0,rep(0,7)); length(beta)

y <- as.matrix(x_onehot) %*% beta  + rnorm(n)

# model setup
inp <- layer_input(shape=c(length(beta)), 
                   batch_shape = NULL)
# deepregression::tibgroup_layer
out <- tibgroup_layer(units=1L, group_idx = NULL, la=1)(inp)
mod <- keras_model(inp, out)

mod %>% compile(loss="mse",
                optimizer = tf$optimizers$Adam(learning_rate=1e-2))

mod %>% fit(x=as.matrix(x_onehot), y=y, epochs=250L, 
            verbose=T, view_metrics = F,
            validation_split=0.1)

mod$get_weights()
apply(Reduce("cbind", lapply(mod$get_weights(),c)), MARGIN = 1, prod)

# torch approach fully manual (2x longer than tf)
tibgroup_module <- tibgroup_layer_torch(units = 1, la = 1, input_shape = length(beta),
                                        group_idx = NULL)
tibgroup_module$parameters
optimizer_tibgrouplasso <- optim_adam(tibgroup_module$parameters, lr=1e-2)
input <- torch_tensor(as.matrix(x_onehot))
data_x <- input
data_y <- torch_tensor(y)

batch_size <- 32
num_data_points <- data_y$size(1)
num_batches <- floor(num_data_points/batch_size)
epochs <- 250

for(epoch in 1:epochs){
  l <- c()
  # rearrange the data each epoch
  permute <- torch_randperm(num_data_points) + 1L
  data_x <- data_x[permute]
  data_y <- data_y[permute]
  # manually loop through the batches
  for(batch_idx in 1:num_batches){
    # here index is a vector of the indices in the batch
    index <- (batch_size*(batch_idx-1) + 1):(batch_idx*batch_size)
    
    x_batch <- data_x[index]
    y_batch <- data_y[index]
    
    optimizer_tibgrouplasso$zero_grad()
    output <- tibgroup_module(x_batch)
    l_tib <- nnf_mse_loss(output, y_batch)
    l_tib$backward()
    optimizer_tibgrouplasso$step()
    l <- c(l, l_tib$item())
  }
  
  cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l)))
}
tibgroup_module$parameters
cbind(
  as.array(mod$weights[[1]] * mod$weights[[2]]),
  as.array((tibgroup_module$parameters[[1]] * tibgroup_module$parameters[[2]])$t()))

# gglasso Ansatz
gr <-  gglasso(model.matrix(~ ., x), as.vector(y), lambda = 1,
               loss="ls", intercept = F)
gr$beta

# now with deepregression
# la gets multiplied by 1/nrow(). So i multipy by 1000 to get a lambda of 1
gr_lasso_mod_tf <- deepregression(y = y, data = x,
                         list_of_formulas = 
                           list(
                             location = ~ 1 + grlasso(x, la  = 1000), 
                             # controls$sp_scale
                             scale = ~1),
                         orthog_options = orthog_control(orthogonalize = F),
                         engine = "tf")

gr_lasso_mod_tf %>% fit(epochs = 250, validation_split = 0.2,
                        early_stopping = F)
gr_lasso_mod_torch <- deepregression(y = y, data = x,
                         list_of_formulas = 
                           list(
                             location = ~ 1 + grlasso(x, la  = 1000),
                             scale = ~1),
                         orthog_options = orthog_control(orthogonalize = F),
                         engine = "torch",
                         subnetwork_builder = subnetwork_init_torch,
                         model_builder = torch_dr)
gr_lasso_mod_torch %>% fit(epochs = 250, validation_split = 0.1, early_stopping = T)

cbind((gr_lasso_mod_tf %>% coef())[[1]],
      (gr_lasso_mod_torch %>% coef())[[1]])
