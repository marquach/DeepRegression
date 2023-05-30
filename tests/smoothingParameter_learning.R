library(torch)

#### GAM data mcycles
data(mcycle, package = 'MASS')
plot(mcycle$times, mcycle$accel)
gam_mgcv <- gam(mcycle$accel ~ 1 + s(times), data = mcycle, method = "GCV.Cp"
                #,sp = 1000000
                )

input_matrix <- model.matrix(gam_mgcv)
input_matrix <- torch_tensor(as.matrix(input_matrix[,-1]))

spline_layer <- nn_linear(in_features = 9, 1)
lambda <- nn_parameter(x = torch_tensor(runif(1)))
P <- torch_tensor(diag(x = 1, 9))

spline_layer$parameters$weight$register_hook(function(grad){
  grad + lambda*torch_matmul((P+P$t()), spline_layer$weight$t())$t()
})

optimizier_spline <- optim_adam(params = spline_layer$parameters, lr = 0.1)
optimizier_lambda <- optim_adam(params = lambda)
epochs <- 1000

for (epoch in 1:epochs) {
  
  optimizier_spline$zero_grad()
  optimizier_lambda$zero_grad()

  
  output <- spline_layer(input_matrix)$view(133)
  loss <- nnf_mse_loss(output, torch_tensor(mcycle$accel))

  if( (epoch%%100) == 0){
    loss <- nnf_mse_loss(output, torch_tensor(mcycle$accel)) +
      lambda * spline_layer$weight$matmul(spline_layer$weight$t())
  }
  
  loss$backward()
  optimizier_spline$step()
  optimizier_lambda$step()
}

spline_layer$parameters$weight
lambda

mean((mcycle$accel-gam_mgcv$fitted.values)^2)

plot(mcycle$times, mcycle$accel)
points(x = mcycle$times, gam_mgcv$fitted.values, col = "red")
points(x = mcycle$times, spline_layer(input_matrix)$view(133), col = "green")
loss
