# simplyconnected_layer_torch Verstehen
library(torch)
library(luz)
source("scripts/deepregression_functions.R")
devtools::load_all("deepregression-main/")
# Tib linear lasso

# benutz Kombitation aus Layern
# simply connected mit l2 loss 
# und dense layer ebenfalls l2 loss



set.seed(42)
n <- 1000
p <- 5
x <- data.frame(runif(n*p) %>% matrix(ncol=p))
y <- 10*x[,1] + rnorm(n)

tib_x1 <- tib_layer_torch(1, 0.1, 1)
tib_x2 <- tib_layer_torch(1, 0.1, 1)


submodules <- list(tib_x1, tib_x2)
test_model <- model_torch(submodules_list = submodules)

test_model()$forward

x_input <- as.list(x[,1:2])
x_test <- get_luz_dataset(x_input, target = y)
test_model$debug("forward")

test_model()$forward(list(x_test$df_list, x_test$target))

optimizer <- optim_adam(params = test_model()$parameters)
for (t in 1:2000) {
  ### -------- Forward pass -------- 
  y_pred <- 
    test_model()(x_input)
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


round(as.array(Reduce(f = "prod", test_model()$parameters[1:2])))
round(as.array(Reduce(f = "prod", test_model()$parameters[3:4])))
round(as.array(Reduce(f = "prod", test_model()$parameters[5:6])))
round(as.array(Reduce(f = "prod", test_model()$parameters[7:8])))
round(as.array(Reduce(f = "prod", test_model()$parameters[9:10])))

set.seed(42)
n <- 1000
p <- 5
x <- data.frame(runif(n*p) %>% matrix(ncol=p))
y <- 2*x[,1] + 3*x[,2] + rnorm(n)
colnames(x) <- paste("x", 1:5, sep = "")

lm(y ~., data = x)

debugonce(deepregression)

lasso_test_torch <- deepregression(y = matrix(y), list_of_formulas = list(
  loc = ~ -1 + lasso(x1) + lasso(x2) + lasso(x3) + lasso(x4) + lasso(x5),
  scale = ~ 1), data = x, engine = "torch",
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  orthog_options = orthog_control(orthogonalize = F)
)
sin_lasso_test_torch <- deepregression(y = matrix(y), list_of_formulas = list(
  loc = ~ -1 + x1 + x2 + x3 + x4 + x5,
  scale = ~ 1), data = x, engine = "torch",
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  orthog_options = orthog_control(orthogonalize = F)
)
lasso_test_tf <- deepregression(y = matrix(y), list_of_formulas = list(
  loc = ~ -1 + lasso(x1) + lasso(x2) + lasso(x3) + lasso(x4) + lasso(x5),
  scale = ~ 1), data = x, engine = "tf",
  orthog_options = orthog_control(orthogonalize = F)
)
debugonce(fit)
lasso_test_torch %>% fit(epochs = 100, early_stopping = T, validation_split = 0.1)
lasso_test_torch %>% coef()

sin_lasso_test_torch %>% fit(epochs = 120, early_stopping = F, 
                             validation_split = 0.1)
sin_lasso_test_torch %>% coef()

lasso_test_tf %>% fit(epochs = 120, early_stopping = F, validation_split = 0)


cbind(lasso_test_tf %>% coef(),
lasso_test_torch %>% coef())


lasso_test_torch <- deepregression(y = matrix(y), list_of_formulas = list(
  loc = ~ -1 + lasso(x1, la = 0.1) + lasso(x2, la = 0.1) + lasso(x3, la = 0.1) +
    lasso(x4, la = 0.1) + lasso(x5, la = 0.1),
  scale = ~ 1), data = x, engine = "torch",
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  orthog_options = orthog_control(orthogonalize = F)
)
lasso_test_tf <- deepregression(y = matrix(y), list_of_formulas = list(
  loc = ~ -1 + lasso(x1, la = 1) + lasso(x2, la = 1) + lasso(x3, la = 1) +
    lasso(x4, la = 1) + lasso(x5, la = 1),
  scale = ~ 1), data = x, engine = "tf",
  orthog_options = orthog_control(orthogonalize = F)
)

sin_lasso_test_tf <- deepregression(y = matrix(y), list_of_formulas = list(
  loc = ~ -1 + x1 + x2 + x3 + x4 + x5,
  scale = ~ 1), data = x, engine = "tf",
  orthog_options = orthog_control(orthogonalize = F)
)

lasso_test_tf %>% fit(epochs = 500, early_stopping = T, validation_split = 0.2)
sin_lasso_test_tf %>% fit(epochs = 100, early_stopping = F, validation_split = 0.2)

lasso_test_tf %>% coef()
sin_lasso_test_tf %>% coef()


lasso_test_torch %>% fit(epochs = 500, early_stopping = F, validation_split = 0.2)
lasso_test_torch %>% coef()

torch_multiply(
  lasso_test$model()$parameters[1:2][[1]],
  lasso_test$model()$parameters[1:2][[2]])
round(as.array(Reduce(f = "prod", lasso_test$model()$parameters[3:4])))
round(as.array(Reduce(f = "prod", lasso_test$model()$parameters[5:6])))
round(as.array(Reduce(f = "prod", lasso_test$model()$parameters[7:8])))



debugonce(deepregression)
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

rigde_test_tf %>% fit(epochs = 50, early_stopping = F, validation_split = 0.2)
rigde_test_tf %>% coef()

rigde_test_torch %>% fit(epochs = 50, early_stopping = F, validation_split = 0.2)
rigde_test_torch %>% coef()






