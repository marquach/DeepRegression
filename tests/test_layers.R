# Script contains tests from  
# https://github.com/neural-structured-additive-learning/deepregression/blob/main/tests/testthat/test_layers.R
# These test the implementation of the lasso layer and their resulting sparsity

library(tensorflow)
library(torch)
library(luz)
library(tictoc)
library(glmnet)

source("scripts_new/deepregression_functions.R")
devtools::load_all("deepregression-main/")

# TF is used as benchmark as this was already implemented
set.seed(42)
tf$random$set_seed(42)
n <- 1000
p <- 50
x <- runif(n*p) %>% matrix(ncol=p)
y <- 10*x[,1] + rnorm(n)

inp <- layer_input(shape=c(p), 
                   batch_shape = NULL)
# deepregression::tib_layer
out <- tib_layer(units=1L, la=1)(inp)
mod <- keras_model(inp, out)

mod %>% compile(loss="mse",
                optimizer = tf$optimizers$Adam(learning_rate=1e-2))
tic()
mod %>% fit(x=x, y=matrix(y), epochs=250L, 
            verbose=T, view_metrics = F,
            validation_split=0.1)
toc()

mod$get_weights()
apply(Reduce("cbind", lapply(mod$get_weights(),c)), MARGIN = 1, prod)

# torch approach fully manual (2x longer than tf)
tib_module <- tib_layer_torch(units = 1, la = 1, input_shape = 50)
optimizer_tiblasso <- optim_adam(tib_module$parameters, lr=1e-2)
input <- torch_tensor(x)
data_x <- input
data_y <- torch_tensor(y)

batch_size <- 32
num_data_points <- data_y$size(1)
num_batches <- floor(num_data_points/batch_size)
tic()
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
    
    optimizer_tiblasso$zero_grad()
    output <- tib_module(x_batch) %>% torch_flatten()
    l_tib <- nnf_mse_loss(output, y_batch)
    l_tib$backward()
    optimizer_tiblasso$step()
    l <- c(l, l_tib$item())
  }
  
  cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l)))
}
toc()

lasso_glmnet <- glmnet(intercept = F, x, y, alpha = 1, lambda  = 2)

cbind("lasso_torch" =  apply(
  cbind(t(as.array(tib_module$parameters[[1]])),
        as.array(tib_module$parameters[[2]])), MARGIN = 1, prod),
      "lasso_tf" = apply(
        Reduce("cbind", lapply(mod$get_weights(),c)), MARGIN = 1, prod),
  "lasso_glmnet" =  coef(lasso_glmnet)[-1])

# looks like torch is able to capture only true relationship better then tf
# keep in mind that lasso is biased
expect_true(abs(Reduce("prod", lapply(tib_module$parameters,
                                      function(x) as.array(x)))) < 1)

# Ridge Ansatz
out <- layer_dense(units=1L, use_bias = F,
                   kernel_regularizer = keras$regularizers$l2(l=0.01))(inp)

mod <- keras_model(inp, out)
mod %>% compile(loss="mse",
                optimizer = tf$optimizers$Adam(learning_rate=1e-2))
mod %>% fit(x=x, y=matrix(y), epochs=500L, 
            verbose=T, view_metrics = F,
            validation_split=0.1)
mod$get_weights()




# torch approach fully manual (2x longer than tf)
ridge_layer <- layer_dense_torch(input_shape = 50, use_bias = F, 
  kernel_regularizer = list( regularizer = "l2", la = 0.01))
optimizer_ridge <- optim_adam(ridge_layer$parameters, lr=1e-2)

tic()
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
    
    optimizer_ridge$zero_grad()
    output <- ridge_layer(x_batch) %>% torch_flatten()
    l_ridge <- nnf_mse_loss(output, y_batch)
    l_ridge$backward()
    optimizer_ridge$step()
    l <- c(l, l_ridge$item())
  }
  
  cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l)))
}
toc()

ridge_glmnet <- glmnet(intercept = F, x, y, alpha = 0, lambda  = 0.1)

data.frame("ridge_torch" =  as.array(ridge_layer$parameters$weight$t()),
      "ridge_tf" = mod$get_weights()[[1]],
      "ridge_glmnet" =  coef(ridge_glmnet)[-1])


# Now with deepregression function
set.seed(42)
n <- 1000
# here only 5 variables
p <- 5
x <- data.frame(runif(n*p) %>% matrix(ncol=p))
y <- 10*x[,1] + rnorm(n)
colnames(x) <- paste("x", 1:5, sep = "")

lasso_test_torch <- deepregression(y = matrix(y), list_of_formulas = list(
  loc = ~ -1 + lasso(x1, la = 1) + lasso(x2, la = 1) + lasso(x3, la = 1) +
    lasso(x4, la = 1) + lasso(x5, la = 1),
  scale = ~ 1), data = x,
  engine = "torch",
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  orthog_options = orthog_control(orthogonalize = F)
)
lasso_test_tf <- deepregression(y = matrix(y), list_of_formulas = list(
  loc = ~ -1 + lasso(x1, la = 1) + lasso(x2, la = 1) + lasso(x3, la = 1) +
    lasso(x4, la = 1) + lasso(x5, la = 1),
  scale = ~ 1), data = x, engine = "tf",
  orthog_options = orthog_control(orthogonalize = F)
)

# adapt learning rate to converge faster
lasso_test_torch$model <- lasso_test_torch$model  %>% set_opt_hparams(lr = 1e-2)
lasso_test_tf$model$optimizer$lr <- tf$Variable(1e-2, name = "learning_rate")

lasso_test_torch %>% fit(epochs = 100, early_stopping = T,
                         validation_split = 0.1, batch_size = 256,
                         fast_fit = F)
lasso_test_tf %>% fit(epochs = 1000, early_stopping = F, validation_split = 0.1,
                      batch_size = 256)

cbind("tf" = lasso_test_tf %>% coef(),
      "torch" = lasso_test_torch %>% coef())

rigde_test_tf <- deepregression(y = matrix(y), list_of_formulas = list(
  loc = ~ -1 + ridge(x1, la = 0.1) + ridge(x2, la = 0.1) + ridge(x3, la = 0.1) +
    ridge(x4, la = 0.1) + ridge(x5, la = 0.1),
  scale = ~ 1), data = x, engine = "tf",
  orthog_options = orthog_control(orthogonalize = F)
)
rigde_test_torch <- deepregression(y = matrix(y), list_of_formulas = list(
  loc = ~ -1 + ridge(x1, la = 0.1) + ridge(x2, la = 0.1) + ridge(x3, la = 0.1) +
    ridge(x4, la = 0.1) + ridge(x5, la = 0.1),
  scale = ~ 1), data = x, engine = "torch",
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  orthog_options = orthog_control(orthogonalize = F)
)

rigde_test_torch$model <- rigde_test_torch$model  %>% set_opt_hparams(lr = 1e-2)
rigde_test_tf$model$optimizer$lr <- tf$Variable(1e-2, name = "learning_rate")

rigde_test_tf %>% fit(epochs = 500, early_stopping = T, validation_split = 0.2)
rigde_test_torch %>% fit(epochs = 500, early_stopping = T, validation_split = 0.2)
ridge_glmnet <- glmnet(intercept = F, x, y, alpha = 0, lambda  = 0.1)

cbind("ridge_tf" = rigde_test_tf %>% coef(),
      "ridge_torch" = rigde_test_torch %>% coef(),
      "ridge_glmnet" = coef(ridge_glmnet)[-1])





