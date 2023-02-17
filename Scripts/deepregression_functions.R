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

# Funktion für Modelbuilder
torch_dr <- function(
    list_pred_param,
    optimizer = torch::optim_adam,
    model_fun = distribution_learning,
    monitor_metrics = list(),
    from_preds_to_output = from_preds_to_dist,
    loss = from_dist_to_loss(family = list(...)$family,
                             weights = weights),
    additional_penalty = NULL,
    ...
){
  outputs <- lapply(list_pred_param, torch_model)
  # define model
  model <- model_fun(outputs, ...)
  
  # allow for optimizer as a function of the model
  if(is.function(optimizer)){
    optimizer <- optimizer(model)
  }
  # compile model
  model %>% luz::setup(optimizer = optimizer,
                    loss = loss,
                    metrics = monitor_metrics)
  
  return(model)
  
}


distribution_learning <- function(neural_net_list, family, output_dim){
  nn_module(
    "distribution_learning_module",
    initialize = function() {
      
      self$distr_parameters <- nn_module_list(
        lapply(lapply(neural_net_list, torch_model), function(x) x()))
    },
    
    forward = function(dataset_list) {
      distribution_parameters <- lapply(
        1:length(self$distr_parameters), function(x){
          self$distr_parameters[[x]](dataset_list[[x]])
        })
      
      dist_fun <- family_to_trochd(family)
      
      # check which distributions are already implemented
      #distr_normal(distribution_parameters[[1]], distribution_parameters[[2]])
      # ACHTUNG: Hier fehlt noch torch_exp bei sigma
      # Je nach Verteilung muss exp gemacht werden
      # Bei expo z.b. Gamma
      #distribution_parameters <- prepare_parameters(family, distribution_parameters)
      #distribution_parameters[[2]] <- torch_exp(distribution_parameters[[2]])
      
      do.call(family, distribution_parameters)
    },
    
    #loss = function(input, target){
    #  torch_mean(-input$log_prob(target))
    #}
  )
}




family_to_trochd <- function(family){
  # define dist_fun
  torchd_dist <- switch(family,
                        normal = distr_normal,
                        bernoulli = distr_bernoulli, 
                        gamma = distr_gamma,
                        poisson = distr_poisson)
}

family_to_trafo <- function(family, add_const = 1e-8){
  
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

from_dist_to_lossfunction <- function(family, weights = NULL){
  
  # define weights to be equal to 1 if not given
  if(is.null(weights)) weights <- 1
  
  # the negative log-likelihood is given by the negative weighted
  # log probability of the dist
  if(family != "normal") stop("Only normal distribution implemented")
  negloglik <- function(input, target)
        -weights*torch_mean(-input$log_prob(target))
  negloglik
  }
  
#from_preds_to_dist_torch
# first part skipped
# then amount of parameters
# check family
#  from_family_to_distfun
# make_tfd_dist:
#   - family_to_trafo parameter properties (Normal: mu: x and sigma torch_exp(...))
#   - create_family(tfd_dist, trafo_list, output_dim)
# create_family:
#   - 


make_torch_dist <- function(family, add_const = 1e-8, output_dim = 1L,
                          trafo_list = NULL)
{
  
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
                 "zipf"
  ) | grepl("multivariate", family) | grepl("vector", family))
  stop("Family ", family, " not implemented yet.")
  
  if(family=="binomial")
    stop("Family binomial not implemented yet.",
         " If you are trying to model independent",
         " draws from a bernoulli distribution, use family='bernoulli'.")
  
  if(is.null(trafo_list)) trafo_list <- family_to_trafo(family)
  
  # check if still NULL, then probably wrong family
  if(is.null(trafo_list))
    stop("Family not implemented.")
  
  ret_fun <- create_family(torch_dist, trafo_list, output_dim)
  
  attr(ret_fun, "nrparams_dist") <- length(trafo_list)
  
  return(ret_fun)
  
}


create_family <- function(torch_dist, trafo_list, output_dim = 1L)
{
  
  if(length(output_dim)==1){
    
    # the usual  case    
    ret_fun <- function(x) do.call(torch_dist,
                                   lapply(1:length(x),
                                          function(i)
                                            trafo_list[[i]](x[[i]]))
    ) 
    
  }else{
    
    # tensor-shaped output (assuming the last dimension to be 
    # the distribution parameter dimension if tfd_dist has multiple arguments)
    dist_dim <- length(trafo_list)
    ret_fun <- function(x) do.call(tfd_dist,
                                   lapply(1:(x$shape[[length(x$shape)]]/dist_dim),
                                          function(i)
                                            trafo_list[[i]](
                                              tf_stride_last_dim_tensor(x,(i-1L)*dist_dim+1L,
                                                                        (i-1L)*dist_dim+dist_dim)))
    ) 
    
  }
  
  attr(ret_fun, "nrparams_dist") <- length(trafo_list)
  
  return(ret_fun)
  
}






from_preds_to_dist <- function(
    list_pred_param,
    family = NULL,
    output_dim = 1L,
    mapping = NULL,
    from_family_to_distfun = make_torch_dist,
    from_distfun_to_dist = distfun_to_dist,
    add_layer_shared_pred = function(x, units) layer_dense(x, units = units,
                                                           use_bias = FALSE),
    trafo_list = NULL
){
  
  nr_params <- length(list_pred_param)
  
  # check family
  if(!is.null(family)){
    if(is.character(family)){
      if(family %in% c("betar", "gammar", "pareto_ls", "inverse_gamma_ls")){
        
        dist_fun <- family_trafo_funs_special(family)
        
      }else{
        
        dist_fun <- from_family_to_distfun(family, output_dim = output_dim,
                                           trafo_list = trafo_list)
        
      }
    }else{ # assuming that family is a dist_fun already
      
      dist_fun <- family
      
    }
  }else{
    
    return(layer_concatenate_identity(unname(list_pred_param)))
    
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
  
  # concatenate predictors
  preds <- layer_concatenate_identity(unname(list_pred_param))
  
  # generate output
  out <- from_distfun_to_dist(dist_fun, preds)
  
  return(out)
  
}






