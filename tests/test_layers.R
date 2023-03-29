library(tensorflow)
library(torch)
devtools::load_all("deepregression-main/")
source("scripts/deepregression_functions.R")

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

mod %>% fit(x=x, y=matrix(y), epochs=500L, 
            verbose=T, view_metrics = F,
            validation_split=0.1)

apply(Reduce("cbind", lapply(mod$get_weights(),c)), MARGIN = 1, prod)

# torch approach
tib_module <- tib_layer_torch(units = 1, la = 1, input_shape = 50)

test_ds <- get_luz_dataset(list(list(x)), y)
test_dl <- test_ds %>% dataloader(batch_size = 32)

iter <- test_dl$.iter()
b <- iter$.next()
optimizer <- optim_adam(tib_module$parameters, lr = 1e-2)

for (epoch in 1:500L) {
  l <- c()
  coro::loop(for (b in test_dl) {
    optimizer$zero_grad()
    output <- tib_module(b[[1]][[1]][[1]]) %>% torch_flatten()
    loss <- nnf_mse_loss(output, b[[2]])
    loss$backward()
    optimizer$step()
    l <- c(l, loss$item())
  })}

apply(cbind(t(as.array(tib_module$parameters[[1]])),
      as.array(tib_module$parameters[[2]])), MARGIN = 1, prod)

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

ridge_layer$parameters


