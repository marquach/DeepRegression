source("Scripts/deepregression_functions.R")
library(ggplot2)
library(mgcv)
skip <- T  # skips calculation of mcycle calculation with luz

# kurzer Exkurs wegen penalty matrix
m1 <- gam(accel ~ -1 + s(times), data = mcycle, family = gaussian)
gam_splines <- model.matrix(m1)
m1$sp
round(m1$smooth[[1]]$S[[1]], 2)
# P Matrix müsste bei deepregression so aussehen
round((0*(m1$smooth[[1]]$S[[1]])),2) # lambda wird bei DRO() auf 0 gesetzt
# bei dr_test$loc[[1]]$penalty wird lambda * P gerechnet
formula <- ~ -1 + s(times)
dr_test <- deepregression(y = accel,
               list_of_formulas = list(
                 loc = formula ,scale = ~ 1), data = mcycle, return_prepoc = T)
dr_test$loc[[1]]$penalty$values
# Exkurs Ende

# Nutze veränderte layer_spline funktion
# Setup nn
torch::torch_manual_seed(42)
test_net <- layer_spline_torch(units = 9, P = torch_zeros(c(9,9)))

grads_self <- ( -2*(t(gam_splines) %*% mcycle$accel) + 2* (
  (t(gam_splines) %*% (gam_splines)) %*% 
    (as.matrix(test_net$parameters$weight$view(9)))))

value <- test_net(gam_splines)$view(length(mcycle$accel))
loss <- nnf_mse_loss(input = value, target = torch_tensor(mcycle$accel),
                     reduction = 'sum')
loss$backward()
cat("Gradient is: ", as.matrix(test_net$parameters$weight$grad), "\n\n")
# now check with self calculated gradients
cat("Gradient is: ", as.matrix(grads_self), "\n\n")
#same

# now test with l2 penalty

# l2 penalty is theta\top %*% theta (reg_constant missing)
# => d/dtheta is 2*theta or (I + I\top)*theta

torch::torch_manual_seed(42)
test_net <- layer_spline_torch(units = 9, P = torch_tensor(diag(9)))

grads_self <- ( -2*(t(gam_splines) %*% mcycle$accel) + 
                  2*( (t(gam_splines) %*% (gam_splines)) %*% 
                  (as.matrix(test_net$parameters$weight$view(9)))))
grad_penalisation <-   2*test_net$parameters$weight$t()

value <- test_net(gam_splines)$view(length(mcycle$accel))
loss <- nnf_mse_loss(input = value, target = torch_tensor(mcycle$accel), 
                     reduction = 'sum')

loss$backward()
cat("Gradient is: ", as.matrix(test_net$parameters$weight$grad), "\n\n")
cat("Gradient is: ", as.matrix(grads_self + grad_penalisation), "\n\n")

if(skip){
# now with luz and dataloader
# Should work because penalty matrix is in layer defined

spline_layer <- nn_module(
  classname = "spline_module", 
  
  initialize = function() {
      self$spline <- layer_spline_torch(units = 9, P = torch_zeros(c(9,9)))
    # Hier keine Penalisierung da torch_zeros
  },
  
  forward = function(x) {
    self$spline(x)}
)

torch_manual_seed(42)
submodules <- list(spline_layer())
# torch_model ist wie keras_model
neural_net <- torch_model(submodules)
neural_net()$forward # Schaut gut aus
neural_net()$parameters # Schaut gut aus

input <- list(torch_tensor(model.matrix(m1)))
luz_dataset <- get_luz_dataset(input, target  = torch_tensor(mcycle$accel))

train_dl <- dataloader(luz_dataset, batch_size = 133, shuffle = F)

pre_fitted <- neural_net %>% 
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
  ) %>% 
  set_opt_hparams(lr = 0.1) 

fitted <- pre_fitted %>% fit(
  data = train_dl, epochs = 1000)

mean((mcycle$accel - m1$fitted.values)^2)

plot(mcycle$times, mcycle$accel)
lines(mcycle$times, m1$fitted.values, col = "red")
lines(mcycle$times, fitted$model(input), col = "green")
# sollte passen
}

# Example deepregression
#get data
set.seed(42)
n <- 1000

data = data.frame(matrix(rnorm(4*n), c(n,4)))
colnames(data) <- c("x1","x2","x3","xa")
y <- rnorm(n) + data$xa^2 + data$x1

# Test
gam_model <- mgcv::gam(y ~ 1+ x1 + s(xa), data = data, family = gaussian)
gam_model$coefficients

torch_manual_seed(42)

spline_layer <- nn_module(
  classname = "spline_module", 
  
  initialize = function() {
    self$spline <- layer_spline_torch(units = 9, P = torch_zeros(c(9,9)))
    # Hier keine Penalisierung da torch_zeros
  },
  
  forward = function(x) {
    self$spline(x)}
)

intercept_layer <- nn_module(
  classname = "intercept_module",
  initialize = function() {
    self$intercept <- nn_linear(1, 1, F) # Bias
  },
  forward = function(x) {
    self$intercept(x)
  }
)

x1_layer <- nn_module(
  classname = "x1_module",
  initialize = function() {
    self$weightx1 <- nn_linear(in_features = 1, out_features = 1, bias = F)
  },
  forward = function(x) {
    self$weightx1(x)
  }
)

submodules <- list(spline_layer(),
                   intercept_layer(),
                   x1_layer()
)

# torch_model ist wie keras_model
neural_net <- torch_model(submodules)
neural_net()$forward # Schaut gut aus
neural_net()$parameters # Schaut gut aus

# hi3r dann überall data_trafo() nutzen
input1 <- torch_tensor(model.matrix(gam_model)[,-c(1,2)]) 
input2 <- torch_tensor(rep(1, 1000))$view(c(1000, 1))
input3 <- torch_tensor(data$x1)$view(c(1000, 1))
inputs_list <- list(input1, input2, input3)

luz_dataset <- get_luz_dataset(inputs_list, target  = torch_tensor(y))

luz_dataset$.getitem(1) # Schaut gut aus, da Target [[2]] ist. 
# Das setzt fit von luz voraus.
train_dl <- dataloader(luz_dataset, batch_size = 1000, shuffle = F)
pre_fitted <- neural_net %>% 
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
  ) %>% 
  set_opt_hparams(lr = 0.1, weight_decay=0.01) 

fitted <- pre_fitted %>% fit(
  data = train_dl, epochs = 200)

plot(gam_model$fitted.values, fitted %>% predict(train_dl),
     xlab = "GAM", ylab = "Neural Net")
abline(a = 0, b = 1, lty = 2)  

plot(gam_model)
points(data$xa,
       model.matrix(gam_model)[,-c(1,2)]%*%t(as.matrix(neural_net()$parameters[[1]])),
       col="red")


# Jetzt mit Weight decay (weight_decay = 0.01)
torch_manual_seed(42)

spline_layer <- nn_module(
  classname = "spline_module", 
  
  initialize = function() {
    self$spline <- layer_spline_torch(units = 9, 
                                      P = torch_tensor(0.01*diag(9)))
    # Hier Identity als Penalty Matrix (sollte ridge-pen. entsprechen)
    # 0.01 ist reg-constant
  },
  
  forward = function(x) {
    self$spline(x)}
)

submodules <- list(spline_layer(),
                   intercept_layer(),
                   x1_layer()
)

# torch_model ist wie keras_model
neural_net <- torch_model(submodules)
neural_net()$forward # Schaut gut aus
neural_net()$parameters # Schaut gut aus

train_dl <- dataloader(luz_dataset, batch_size = 1000, shuffle = F)
pre_fitted <- neural_net %>% 
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
  ) %>% 
  set_opt_hparams(lr = 0.1) 

fitted <- pre_fitted %>% fit(
  data = train_dl, epochs = 200)
points(data$xa,
       model.matrix(gam_model)[,-c(1,2)]%*%t(as.matrix(neural_net()$parameters[[1]])),
       col="green")
