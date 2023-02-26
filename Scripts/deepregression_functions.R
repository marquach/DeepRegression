#' Function to define a torch layer similar to a tf dense layer
#' 
#' @param units integer; number of output units
#' @param P matrix; penalty matrix
#' @param name string; string defining the layer's name
#' @param trainable logical; whether layer is trainable
#' @param kernel_initializer initializer; for basis coefficients
#' @return Torch layer
#' @export
layer_dense_torch <- function(units, name, trainable = TRUE,
                         kernel_initializer = "glorot_uniform"){
  
  layer_torch <- nn_linear(units, out_features = 1, bias = FALSE)
  
  if (kernel_initializer == "glorot_uniform") {
    nn_init_xavier_uniform_(
      tensor = layer_torch$weight,
      gain = nn_init_calculate_gain(nonlinearity = "linear"))
  }
  
  if(!trainable) layer_torch$parameters$weight$requires_grad = FALSE
  
  layer_torch
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
  
  P <- torch_tensor(P)
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


model_torch <-  function(submodules_list){
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
    indexes <- lapply(self$df_list,
                      function(x) lapply(x, function(x) x[index,]))
    
    target <- self$target[index]$view(1)
    list(indexes, target)
  },
  
  .length = function() {
    length(self$target)
  }
  
)


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
    
      outputs <- lapply(1:length(pp_in),
                        function(i) pp_lay[[layer_matching[i]]]$layer())
      return(outputs) 
  
  }
}

# Funktion für Modelbuilder
torch_dr <- function(
    list_pred_param,
    optimizer = torch::optim_adam,
    model_fun = distribution_learning,
    monitor_metrics = list(),
    from_preds_to_output = from_preds_to_dist_torch,
    loss = from_dist_to_loss_torch(family = list(...)$family,
                             weights = weights),
    additional_penalty = NULL,
    ...
){

  out <- from_preds_to_output(list_pred_param, ...)
  # define model
  model <- out
  
  # compile model
  model <- model %>% luz::setup(optimizer = optimizer,
                    loss = loss,
                    metrics = monitor_metrics)
  
  return(model)
  
}

from_preds_to_dist_torch <- function(
    list_pred_param, family = NULL,
    output_dim = 1L,
    mapping = NULL,
    from_family_to_distfun = make_torch_dist,
    from_distfun_to_dist = distfun_to_dist_torch,
    #add_layer_shared_pred = function(x, units) layer_dense_torch(x, units = units,
    #                                                       use_bias = FALSE),
    trafo_list = NULL){
  
  nr_params <- length(list_pred_param)
  
  # check family
  if(!is.null(family)){
    if(is.character(family)){
      dist_fun <- from_family_to_distfun(family, output_dim = output_dim,
                                         trafo_list = trafo_list)
    }
  } else{ # assuming that family is a dist_fun already
    dist_fun <- family
  } 
  
  nrparams_dist <- attr(dist_fun, "nrparams_dist")
  if(is.null(nrparams_dist)) nrparams_dist <- nr_params
  
  if(nrparams_dist < nr_params)
  {
    warning("More formulas specified than parameters available.",
            " Will only use ", nrparams_dist, " formula(e).")
    nr_params <- nrparams_dist
    list_pred_param <- list_pred_param[1:nrparams_dist]
  }else if(nrparams_dist > nr_params){
    stop("Length of list_of_formula (", nr_params,
         ") does not match number of distribution parameters (",
         nrparams_dist, ").")
  }
  
  if(is.null(names(list_pred_param))){
    names(list_pred_param) <- names_families(family)
  }
  
  preds <- lapply(list_pred_param, model_torch)
  
  # generate output
  out <- from_distfun_to_dist_torch(dist_fun, preds)
  
}


from_distfun_to_dist_torch <- function(dist_fun, preds){
  nn_module(
    
    initialize = function() {
      
      self$distr_parameters <- nn_module_list(
        lapply(preds, function(x) x()))
    },
    
    forward = function(dataset_list) {
      distribution_parameters <- lapply(
        1:length(self$distr_parameters), function(x){
          self$distr_parameters[[x]](dataset_list[[x]])
        })
      dist_fun(distribution_parameters)
    }
  )
}






make_torch_dist <- function(family, add_const = 1e-8, output_dim = 1L,
                            trafo_list = NULL){
  
  torch_dist <- family_to_trochd(family)
  
  # families not yet implemented
  if(family%in%c("categorical",
                 "dirichlet_multinomial",
                 "dirichlet",
                 "gamma_gamma",
                 "geometric",
                 "kumaraswamy",
                 "truncated_normal",
                 "von_mises",
                 "von_mises_fisher",
                 "wishart",
                 "zipf") | grepl("multivariate", family) | grepl("vector", family))
    stop("Family ", family, " not implemented yet.")
  
  if(family=="binomial")
    stop("Family binomial not implemented yet.",
         " If you are trying to model independent",
         " draws from a bernoulli distribution, use family='bernoulli'.")
  
  if(is.null(trafo_list)) trafo_list <- family_to_trafo_torch(family)
  
  # check if still NULL, then probably wrong family
  if(is.null(trafo_list))
    stop("Family not implemented.")
  
  ret_fun <- create_family_torch(torch_dist, trafo_list, output_dim)
  
  attr(ret_fun, "nrparams_dist") <- length(trafo_list)
  
  return(ret_fun)
  
}

family_to_trochd <- function(family){
  # define dist_fun
  torchd_dist <- switch(family,
                        normal = distr_normal,
                        bernoulli = distr_bernoulli, 
                        gamma = distr_gamma,
                        poisson = distr_poisson)
}

family_to_trafo_torch <- function(family, add_const = 1e-8){
  
  trafo_list <- switch(family,
                       normal = list(function(x) x,
                                     function(x) torch_add(add_const, torch_exp(x))),
                       bernoulli = list(function(x) x),
                       gamma = list(
                         function(x) torch_add(add_const, torch_exp(x)),
                         function(x) torch_add(add_const, torch_exp(x))),
                       poisson = list(function(x) torch_add(add_const, torch_exp(x)))
  )
  
  return(trafo_list)
  
}

create_family_torch <- function(torch_dist, trafo_list, output_dim = 1L)
{
  
  if(length(output_dim)==1){
    
    # the usual  case    
    ret_fun <- function(x) do.call(torch_dist,
                                   lapply(1:length(x),
                                          function(i)
                                            trafo_list[[i]](x[[i]]))
    ) 
    
  }
  #else{
  
  # tensor-shaped output (assuming the last dimension to be 
  # the distribution parameter dimension if tfd_dist has multiple arguments)
  # dist_dim <- length(trafo_list)
  #  ret_fun <- function(x) do.call(tfd_dist,
  #                                lapply(1:(x$shape[[length(x$shape)]]/dist_dim),
  #                                     function(i)
  #                                       trafo_list[[i]](
  #                                        tf_stride_last_dim_tensor(x,(i-1L)*dist_dim+1L,
  #                                                                   (i-1L)*dist_dim+dist_dim)))
  #) 
  
  #}
  
  attr(ret_fun, "nrparams_dist") <- length(trafo_list)
  
  return(ret_fun)
  
}


from_dist_to_loss_torch <- function(family, weights = NULL){
  
  # define weights to be equal to 1 if not given
  #if(is.null(weights)) weights <- 1
  
  # the negative log-likelihood is given by the negative weighted
  # log probability of the dist
  if(family != "normal") stop("Only normal distribution implemented")
  negloglik <- function(input, target) torch_mean(-input$log_prob(target))
  negloglik
  }
  