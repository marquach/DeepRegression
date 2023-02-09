#' Function to define a torch layer similiar to a tf dense layer
#' 
#' @param units integer; number of output units
#' @param P matrix; penalty matrix
#' @param name string; string defining the layer's name
#' @param trainable logical; whether layer is trainable
#' @param kernel_initializer initializer; for basis coefficients
#' @return Torch layer
#' @export
torch_layer_dense <- function(units, name, trainable = TRUE,
                         kernel_initializer = "glorot_uniform"){
  
  torch_layer <- nn_linear(units, out_features = 1, bias = FALSE)
  
  if (kernel_initializer == "glorot_uniform") {
    nn_init_xavier_uniform_(
      tensor = torch_layer$weight,
      gain = nn_init_calculate_gain(nonlinearity = "linear"))
  }
  
  if(!trainable) torch_layer$parameters$weight$requires_grad = FALSE
  
  torch_layer
}



#' Function to define spline as Torch layer
#' 
#' @param units integer; number of output units
#' @param P matrix; penalty matrix
#' @param name string; string defining the layer's name
#' @param trainable logical; whether layer is trainable
#' @param kernel_initializer initializer; for basis coefficients
#' @return Torch layer
#' @export
layer_spline_torch <- function(P, units, name, trainable = TRUE,
                               kernel_initializer = "glorot_uniform"){
  
  spline_layer <- nn_linear(units, out_features = 1, bias = FALSE)
  
  if (kernel_initializer == "glorot_uniform") {
    nn_init_xavier_uniform_(
      tensor = spline_layer$weight,
      gain = nn_init_calculate_gain(nonlinearity = "linear"))
  }
  
  spline_layer$parameters$weight$register_hook(function(grad){
    grad + torch_matmul((P+P$t()), spline_layer$weight$t())$t()
  })
  
  if(!trainable) spline_layer$parameters$weight$requires_grad = FALSE
  
  spline_layer
}


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
  "deepregression_luz_dataset",
  
  initialize = function(df_list, target) {
    self$df_list <- df_list
    self$target <- target
  },
  
  .getitem = function(index) {
    indexes <- lapply(self$df_list, function(x) x[index,])
    target <- self$target[index]$view(1)
    list(indexes, target)
  },
  
  .length = function() {
    length(self$target)
  }
  
)

