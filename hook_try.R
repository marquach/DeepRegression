
# Teste hooks indem gradient auf null gesetzt wird bei register_hook
# Loss sollte dann gleich bleiben, gradient mal 0 genommen wird

library(luz)
library(torch)

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


intercept_layer <- nn_module(
  classname = "intercept_module",
  initialize = function() {
    self$intercept <- nn_linear(1, 1, F) # Bias
  },
  forward = function(x) {
    self$intercept(x)
  }
)

P <- diag(1)

str(test$parameters)
test <- intercept_layer()
h <- lapply(test$parameters, 
            function(x) x$register_hook(function(grad) grad + lambda* P %*% x))
# Für hook muss layer scheinbar schon generiert sein, 
# da register_hook einen Tensor braucht.

submodules <- list(test)
# submodules ist output von subnetwork_init

# torch_model ist wie keras_model
neural_net <- torch_model(submodules)
neural_net()$forward # Schaut gut aus
neural_net()$parameters # Schaut gut aus

# hi3r dann überall data_trafo() nutzen
input <- torch_tensor(rep(1, 1000))$view(c(1000, 1))
inputs_list <- list(input)

luz_dataset <- get_luz_dataset(inputs_list, target  = torch_tensor(y))

luz_dataset$.getitem(1) # Schaut gut aus, da Target [[2]] ist. 
# Das setzt fit von luz voraus.
train_dl <- dataloader(luz_dataset, batch_size = 1000, shuffle = F)
pre_fitted <- neural_net %>% 
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
  ) %>% 
  set_opt_hparams(lr = 0.1) 

fitted <- pre_fitted %>% fit(
  data = train_dl, epochs = 10)

unlist(fitted$records$metrics$train)
# Scheint zu funktionieren, da loss gleich bleibt

# Teste mit glmtnet ob l1 bzw. l2 regu funktioniert
# https://stackoverflow.com/questions/42704283/l1-l2-regularization-in-pytorch
# https://github.com/szymonmaszke/torchlayers/blob/master/torchlayers/regularization.py#L150
# test$parameters$intercept.weight$data()


library(glmnet)
data(QuickStartExample)
x <- QuickStartExample$x;
y <- QuickStartExample$y
lasso_res <- glmnet(x, y, alpha = 1)
coef(lasso_res, s = 1) 
lasso_ridge <- glmnet(x, y, alpha = 0)
coef(lasso_ridge, s = 1) 





