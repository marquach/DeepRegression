#' Function to define a torch layer similar to a tf dense layer
#' 
#' @param units integer; number of output units
#' @param name string; string defining the layer's name
#' @param trainable logical; whether layer is trainable
#' @param kernel_initializer initializer; for coefficients
#' @return Torch layer
#' @export
layer_dense_torch <- function(input_shape, units = 1L, name, trainable = TRUE,
                              kernel_initializer = "glorot_uniform",
                              use_bias = FALSE, kernel_regularizer = NULL){
  
  layer <- nn_linear(in_features = input_shape,
                     out_features = units, bias = use_bias)
  
  if (kernel_initializer == "glorot_uniform") {
    nn_init_xavier_uniform_(
      tensor = layer$weight,
      gain = nn_init_calculate_gain(nonlinearity = "linear"))
  }
  
  if (kernel_initializer == "torch_ones") {
    nn_init_ones_(layer$parameters$weight)
  }
  
  if(!trainable) layer$parameters$weight$requires_grad = FALSE
  
  if(!is.null(kernel_regularizer)){
    
    if(kernel_regularizer$regularizer == "l2") {
      layer$parameters$weight$register_hook(function(grad){
        grad + 2*(kernel_regularizer$la)*(layer$parameters$weight)
      })
    }}
  
  layer
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
layer_spline_torch <- function(P, units = 1L, name, trainable = TRUE,
                               kernel_initializer = "glorot_uniform"){
  
  P <- torch_tensor(P)
  input_shape <- P$size(1)
  spline_layer <- nn_linear(in_features = input_shape,
                            out_features = units, bias = FALSE)
  
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
