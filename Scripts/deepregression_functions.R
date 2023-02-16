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


# Funtkion für Subnetwork Init in torch
# Schreibe einfach selber eine mit endung torch und lade diese dann am Anfang,
# da subnetwork_builder eine Parameter der Funktion ist


subnetwork_init_torch <- function(pp, deep_top = NULL, 
                            orthog_fun = orthog_tf, 
                            split_fun = split_model,
                            shared_layers = NULL,
                            param_nr = 1,
                            selectfun_in = function(pp) pp[[param_nr]],
                            selectfun_lay = function(pp) pp[[param_nr]],
                            gaminputs,
                            summary_layer = layer_add_identity)
{
  
  # instead of passing the respective pp,
  # subsetting is done within subnetwork_init
  # to allow other subnetwork_builder to 
  # potentially access all pp entries
  pp_in <- selectfun_in(pp)
  pp_lay <- selectfun_lay(pp)
  
 
  layer_matching <- 1:length(pp_in)
  names(layer_matching) <- layer_matching
  
  
  if(all(sapply(pp_in, function(x) is.null(x$right_from_oz)))){ 
    # if there is no term to orthogonalize
    
      outputs <- lapply(1:length(pp_in), function(i) pp_lay[[layer_matching[i]]]$layer())
      return(outputs) 
  
  }
}
