#functions

## Folgende Funktion sollte Punkt 2 von TF zu Torch umgewandelt werden:
# def layer_spline(P, units, name, trainable = TRUE,
#                 kernel_initializer = "glorot_uniform"):
#  
#  return(tf.keras.layers.Dense(units = units, name = name, use_bias=False,
#                               kernel_regularizer = squaredPenalty(P, 1),
#                               trainable = trainable,
#                               kernel_initializer = kernel_initializer))

# tf:layers.Dense <==> torch:nn_linear
# Direkt bei nn_linear kein kernel_initializer möglich (nur indirekt: nn_init_)

# kernel_regularizer = squaredPenalty(P, 1) gibt es nicht bei Torch
# Regularisierung direkt bei Loss Berechnung??

# Per hook implementiert, da layer unterschiedlich penalisiert werden können

layer_spline_torch <- function(units = 1L, name, #trainable = TRUE,
                               kernel_initializer = "glorot_uniform",
                         P){
  
  spline_layer <- nn_linear(units, out_features = 1, bias = F)
  
  if (kernel_initializer == "glorot_uniform") {
    nn_init_xavier_uniform_(
      tensor = spline_layer$weight,
      gain = nn_init_calculate_gain(nonlinearity = "linear"))
  }
  
  # nicht sicher ob P immer symmetrisch ist, deswegen P+P$t() statt 2*P
  # glaub aber schon
  spline_layer$parameters$weight$register_hook(function(grad){
    grad + torch_matmul((P+P$t()), spline_layer$weight$t())$t()
  })
  
  
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

# Dataloader müsste dann m.M.n. in deepregression erstellt werden,
# da außerhalb nicht schön wäre bzw. zu sehr von keras abweicht.
# Input wird ja eh schon in Form von data gegeben, also sollte das passen

get_luz_dataset <- dataset(
  "deepregression_luz_dataset",
  
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
  
  # Braucht man für dataloader. Wenn Shuffle aus wird es nicht genutzt.
  # Scheint als ob shuffle so nocht nicht funktioniert. (Später anschauen)
  .length = function() {
    length(self$target)
  }
  
)

