# Test Skript für Hooks

#get data
set.seed(42)
n <- 1000

data = data.frame(matrix(rnorm(4*n), c(n,4)))
colnames(data) <- c("x1","x2","x3","xa")
y <- rnorm(n) + data$xa^2 + data$x1


torch_model <-  function(submodules_list){
  nn_module(
    classname = "torch_model",
    initialize = function() {
      self$subnetworks <- nn_module_list(submodules_list)
    },
    
    forward = function(dataset_list) {
      subnetworks <- lapply(1:length(self$subnetworks), function(x){
        self$subnetworks[[x]](dataset_list[[x]])
      })
      Reduce(f = "+", x = subnetworks)}
  )}


get_luz_dataset <- dataset(
  "mydataset",
  
  initialize = function(df_list, target) {
    self$df_list <- df_list
    self$target <- target
  },
  
  .getitem = function(index) {
    indexes <- lapply(self$df_list, function(x) x[index,])
    target <- self$target[index]$view(1)
    # note that the dataloaders will automatically stack tensors
    # creating a new dimension
    list(indexes, target)
  },
  .length = function() {
    length(self$target)
  }
)


spline_layer <- nn_module(
  classname = "spline_module", 
  
  initialize = function() {
    self$spline <- nn_linear(in_features = 9, out_features = 1, bias = F)
    
    #layer_spline(units = 9, #mod_true_preproc$loc[[1]]$input_dim
    #name = "spline_layer")
  },
  forward = function(x) {
    self$spline(x)}
)

torch::torch_manual_seed(42)
submodules <- list(spline_layer())
# submodules ist output von subnetwork_init
neural_net <- torch_model(submodules)
neural_net()$forward # Schaut gut aus
neural_net()$parameters # Schaut gut aus

grads_self <- 1/(1000) * ( -2*(t(model.matrix(gam_model)[,-c(1,2)]) %*% y) + 2* (
          (t(model.matrix(gam_model)[,-c(1,2)]) %*% (model.matrix(gam_model)[,-c(1,2)])) %*% 
          (as.matrix(neural_net()$parameters$subnetworks.0.spline.weight$view(9)))))

# hi3r dann überall data_trafo() nutzen
input1 <- torch_tensor(model.matrix(gam_model)[,-c(1,2)]) 


for (i in 1:1) {
  value <- neural_net()$forward(list(input1))$view(1000)
  loss <- nnf_mse_loss(input = value, target = torch_tensor(y))
  #neural_net()$parameters$subnetworks.0.spline.weight$register_hook( function(grad) grad*0) # gradient auf null setzen
  
  loss$backward()
  cat("Gradient is: ", as.matrix(neural_net()$parameters$subnetworks.0.spline.weight$grad), "\n\n")
  grads <- as.matrix(neural_net()$parameters$subnetworks.0.spline.weight$grad)
  
  
  # manual update
  with_no_grad({
    neural_net()$parameters$subnetworks.0.spline.weight$sub_(lr * neural_net()$parameters$subnetworks.0.spline.weight$grad)
    neural_net()$parameters$subnetworks.0.spline.weight$grad$zero_()
  })
  
}
grads
t(grads_self)

# Passt

torch::torch_manual_seed(42)
submodules <- list(spline_layer())
# submodules ist output von subnetwork_init
# torch_model ist wie keras_model
neural_net <- torch_model(submodules)
reg_constant = 0.01
for (i in 1:1) {
  value <- neural_net()$forward(list(input1))$view(1000)
  loss <- nnf_mse_loss(input = value, target = torch_tensor(y))
  penalty_term <- 2*neural_net()$parameters$subnetworks.0.spline.weight
  neural_net()$parameters$subnetworks.0.spline.weight$register_hook(
    function(grad) grad + reg_constant*penalty_term) # gradient auf null setzen
  
  loss$backward()
  cat("Gradient is: ",
      as.matrix(neural_net()$parameters$subnetworks.0.spline.weight$grad), "\n\n")
}
grads
penalty_term
grads + 0.01*penalty_term

grads_self*2

neural_net()$parameters$subnetworks.0.spline.weight$grad 
neural_net()$parameters$subnetworks.0.spline.weight
test$parameters$intercept.weight



