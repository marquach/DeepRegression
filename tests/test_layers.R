library(tensorflow)
library(torch)
library(luz)
source("scripts/deepregression_functions.R")
devtools::load_all("deepregression-main/")

set.seed(42)
tf$random$set_seed(42)
n <- 1000
p <- 50
x <- runif(n*p) %>% matrix(ncol=p)
y <- 10*x[,1] + rnorm(n)

inp <- layer_input(shape=c(p), 
                   batch_shape = NULL)
out <- tib_layer(units=1L, la=1)(inp)
mod <- keras_model(inp, out)

mod %>% compile(loss="mse",
                optimizer = tf$optimizers$Adam(learning_rate=1e-2))

mod %>% fit(x=x, y=matrix(y), epochs=250L, 
            verbose=T, view_metrics = F,
            validation_split=0.1)
mod$get_weights()
apply(Reduce("cbind", lapply(mod$get_weights(),c)), MARGIN = 1, prod)

# torch approach
tib_module <- tib_layer_torch(units = 1, la = 1, input_shape = 50)
linear_module <- layer_dense_torch(units = 50, use_bias = F)

test_ds <- get_luz_dataset(df_list = list(list(x)), target = y)
test_dl <- test_ds %>% dataloader(batch_size = 256)

iter <- test_dl$.iter()
b <- iter$.next()
optimizer_tib <- optim_adam(tib_module$parameters, lr = 1e-2)
optimizer_lin <- optim_adam(linear_module$parameters, lr = 1e-2)


for (epoch in 1:250) {

  coro::loop(for (b in test_dl) {
    optimizer_tib$zero_grad()
    optimizer_lin$zero_grad()
    output_tib <- tib_module(b[[1]][[1]][[1]]) %>% torch_flatten()
    output_lin <- linear_module(b[[1]][[1]][[1]]) %>% torch_flatten()
    loss_tib <- nnf_mse_loss(output_tib, b[[2]])
    loss_lin <- nnf_mse_loss(output_lin, b[[2]])
    
    loss_tib$backward()
    loss_lin$backward()
    
    optimizer_tib$step()
    optimizer_lin$step()
    

  })
  cat("Epoch: ", epoch, "   Loss_tib: ", loss_tib$item(),
  "Loss_lin: ", loss_lin$item(),  "\n")
}

cbind("ridge_torch" =  apply(cbind(t(as.array(tib_module$parameters[[1]])),
      "dense_torch" = as.array(tib_module$parameters[[2]])), MARGIN = 1, prod),
      unlist(t(as.array(linear_module$parameters[[1]]))),
      "ridge_tf" = apply(Reduce("cbind", lapply(mod$get_weights(),c)), MARGIN = 1, prod)
)
# looks like torch is able to capture only true relationship better then tf
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

ridge_layer <- layer_dense_torch(units = 50, use_bias = F, 
                  kernel_regularizer = list( regularizer = "l2", la = 0.01))

iter <- test_dl$.iter()
b <- iter$.next()
optimizer <- optim_adam(ridge_layer$parameters, lr = 1e-2)


for (epoch in 1:500L) {
  l <- c()
  coro::loop(for (b in test_dl) {
    optimizer$zero_grad()
    output <- ridge_layer(b[[1]][[1]][[1]]) %>% torch_flatten()
    loss <- nnf_mse_loss(output, b[[2]])
    loss$backward()
    optimizer$step()
    l <- c(l, loss$item())
  })}

cbind( "ridge_torch" = t(as.array(ridge_layer$parameters$weight)),
       "ridge_tf" = mod$get_weights()[[1]])
#almost same

# Now with deepregression function
set.seed(42)
n <- 1000
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
lasso_test_torch %>% fit(epochs = 1000, early_stopping = F,
                         validation_split = 0.1, batch_size = 256)
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

rigde_test_tf %>% fit(epochs = 100, early_stopping = F, validation_split = 0.2)
rigde_test_tf %>% coef()

rigde_test_torch %>% fit(epochs = 100, early_stopping = F, validation_split = 0.2)
rigde_test_torch %>% coef()

solve( t(x) %*% as.matrix(x) + 0.1 * diag(5), t(x) %*%y)

library(glmnet)
ridge_glmnet <- glmnet(intercept = F, x, y, alpha = 0, lambda  = 0.1)
coef(ridge_glmnet)
lasso_glmnet <- glmnet(intercept = F, x, y, alpha = 1, lambda  = 1)
coef(lasso_glmnet)




