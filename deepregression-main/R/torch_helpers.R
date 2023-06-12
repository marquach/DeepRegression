#' Helper function to create an function that generates R6 instances of 
#' class dataset
#' @param df_list list; data for the distribution learning model (data for every distributional parameter)
#' @param target vector; target value
#' @return R6 instances of class dataset
#' @export
get_luz_dataset <- dataset(
  "deepregression_luz_dataset",
  
  initialize = function(df_list, target = NULL, length = NULL) {
    
    self$data <- self$setup_loader(df_list)
    self$target <- target
    if(!is.null(length)) self$length <- length 
    
  },
  
  # has to be fast because is used very often (very sure this a bottle neck)
  .getbatch = function(index) {
    
    indexes <- lapply(self$data,
                      function(x) lapply(x, function(y) y(index)))
    
    if(is.null(self$target)) return(list(indexes))
    
    target <- self$target[index]
    list(indexes, target)
  },
  
  .length = function() {
    if(!is.null(self$length)) return(self$length)
    length(self$target)
    
  },
  
  setup_loader = function(df_list){
    
    lapply(df_list, function(x) 
      lapply(x, function(y){
        if((ncol(y)==1) & check_data_for_image(y)){
          return( 
            function(index) torch_stack(
              lapply(index, function(x) y[x, ,drop = F] %>% base_loader() %>%
                       transform_to_tensor())))
          # this torch_stack(...) allows to use .getbatch also when
          # we use image data.
        }
        function(index) torch_tensor(y[index, ,drop = F])
      }))
  }
)


# till now only orthog_options will be test
# Motivation is to test inputs given engine
check_input_torch <- function(orthog_options){
  if(orthog_options$orthogonalize != FALSE) 
    stop("Orthogonalization not implemented for torch")
}


get_weights_torch <- function(model){
  old_weights <- lapply(model$model()$parameters, function(x) as_array(x))
  lapply(old_weights, function(x) torch_tensor(x))
}
# prepare_input_list_model()



weight_reset <-  function(m) {
  try(m$reset_parameters(), silent = T)
}


collect_distribution_parameters <- function(family){
  parameter_list <- switch(family,
                           normal = function(x) list("loc" = x$loc,
                                                     "scale" = x$scale),
                           bernoulli = function(x) list("logits" = x$logits),
                           bernoulli_prob = function(x) list("probs" = x$probs),
                           poisson = function(x) list("rate" = x$rate),
                           gamma = function(x) list("concentration" = 
                                                      x$concentration,
                                                    "rate" = x$rate))
  parameter_list
}




prepare_torch_distr_mixdistr <- function(object, dists){
  
  helper_collector <- collect_distribution_parameters(object$init_params$family)
  distr_parameters <- lapply(dists, helper_collector)
  num_params <- length(distr_parameters[[1]])
  
  distr_parameters <- lapply(seq_len(num_params),
                             function(y) lapply(distr_parameters,
                                                FUN = function(x) x[[y]]))
  distr_parameters <- lapply(distr_parameters, FUN = function(x) torch_cat(x, 2))
  distr_parameters
}

# check if variable contains image data

check_data_for_image <- function(data){
  jpg_yes <- all(grepl(x = data, pattern = ".jpg"))
  png_yes <- all(grepl(x = data, pattern = ".png"))
  image_yes <- jpg_yes | png_yes
  image_yes
}

choose_kernel_initializer_torch <- function(kernel_initializer, value = NULL){
  kernel_initializer_value <- value
  
  if( kernel_initializer == "constant"){
    kernel_initializer <-  function(value)
      nn_init_no_grad_constant_deepreg(
        tensor = self$weight, value = value)
    formals(kernel_initializer)$value <- kernel_initializer_value
    return(kernel_initializer)
  }
  
  kernel_initializer <- switch(kernel_initializer,
                               "glorot_uniform" = 
                                 function(){
                                   nn_init_xavier_uniform_(tensor = self$weight,
                                                           gain = nn_init_calculate_gain(
                                                             nonlinearity = "linear"))},
                               "torch_ones" = function() 
                                 nn_init_ones_(self$weight),
                               "he_normal" = function()
                                 nn_init_kaiming_normal_(self$weight)
  )
  
  kernel_initializer
}
