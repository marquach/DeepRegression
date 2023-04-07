
simplyconnected_layer_torch <- function(la = la,
                                        multfac_initializer = torch_ones,
                                        input_shape){
  nn_module(
    classname = "simply_con",
    
    initialize = function(){
      self$la <- torch_tensor(la)
      self$multfac_initializer <- multfac_initializer
      
      sc <- nn_parameter(x = self$multfac_initializer(input_shape))
      
      sc$register_hook(function(grad){
        grad + la*sc
      })
      self$sc <- sc
    },
    
    # muss angepasst werden
    # wenn self$sc mehrere sind muss es transponiert werden
    forward = function(dataset_list){
      torch_multiply(
        self$sc$view(c(1, length(self$sc))),
        dataset_list)
      }
  )
}


tiblinlasso_layer_torch <- function(la, input_shape = 1, units = 1,
                                    kernel_initializer = "he_normal"){
  
  la <- torch_tensor(la)
  tiblinlasso_layer <- nn_linear(in_features = input_shape,
                                 out_features = units, bias = F)
  if (kernel_initializer == "he_normal") {
    nn_init_kaiming_normal_(tiblinlasso_layer$parameters$weight)
  }
  
  tiblinlasso_layer$parameters$weight$register_hook(function(grad){
    grad + la*tiblinlasso_layer$parameters$weight
  })
  tiblinlasso_layer
}

tib_layer_torch <-
  nn_module(
    classname = "TibLinearLasso_torch",
    initialize = function(units, la, input_shape, multfac_initializer = torch_ones){
      self$units = units
      self$la = la
      self$multfac_initializer = multfac_initializer
      
      self$fc <- tiblinlasso_layer_torch(la = self$la,
                                         input_shape = input_shape,
                                         units = self$units)
      self$sc <- simplyconnected_layer_torch(la = self$la,
                                             input_shape = input_shape,
                                             multfac_initializer = 
                                               self$multfac_initializer)()
      
    },
    forward = function(data_list){
      self$fc(self$sc(data_list))
    }
  )




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
  
  if(!trainable) layer$parameters$weight$requires_grad = FALSE
  
  if(!is.null(kernel_regularizer)){
    
    if(kernel_regularizer$regularizer == "l2") {
      layer$parameters$weight$register_hook(function(grad){
        grad + (kernel_regularizer$la)*(layer$parameters$weight)
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

#' Function to initialize a nn_module
#' Forward functions works with a list. The entries of the list are the input of 
#' the subnetworks
#' 
#' @param submodules_list list; subnetworks of distribution parameter
#' @return nn_module
#' @export

model_torch <-  function(submodules_list){
  nn_module(
    classname = "torch_model",
    initialize = function() {
      self$subnetwork <- nn_module_list(submodules_list)
      names(self$subnetwork$.__enclos_env__$private$modules_) <- 
        names(submodules_list)
      
    },
    
    forward = function(dataset_list) {
      subnetworks <- lapply(1:length(self$subnetwork), function(x){
        self$subnetwork[[x]](dataset_list[[x]])
      })
      
      #Reduce(f = "+", x = subnetworks)
      torch_sum(torch_stack(subnetworks), dim = 1)
      }
  )}


#' Helper function to create an function that generates R6 instances of 
#' class dataset
#' @param df_list list; data for the distribution learning model (data for every distributional parameter)
#' @param target vector; target value
#' @return R6 instances of class dataset
#' @export
get_luz_dataset <- dataset(
  "deepregression_luz_dataset",
  
  initialize = function(df_list, target = NULL, length = NULL) {
    self$data <- self$prepare_data(df_list)
    
    self$target <- target
    self$length <- length
    
    
  },
  
  .getitem = function(index) {
    
    # am besten hier je nach typ bestimmte methode durchführen
    indexes <- lapply(self$data,
                      function(x) lapply(x, function(y) {
                        if(!inherits(y, "torch_tensor"))
                          if(colnames(y) == "image"){
                          return(y[index,] %>% base_loader() %>%
                                   torchvision::transform_to_tensor())}
                        y[index,]}))
    if(is.null(self$target)) return(list(indexes))
    target <- self$target[index]
    list(indexes, target)
  },
  
  .length = function() {
    if(!is.null(self$length)) return(self$length)
    length(self$target)
    
  },
  
  prepare_data = function(df_list){
    
    lapply(df_list, function(x) 
      lapply(x, function(y){
        if((ncol(y)==1) & check_data_for_image(y)){
          colnames(y) <- "image"
          #print(sprintf("Found %s validated image filenames", length(y)))
          return(y)
          }
        torch_tensor(y)}))
    }
)


# check if variable contains image data

check_data_for_image <- function(data){
  jpg_yes <- all(grepl(x = data, pattern = ".jpg"))
  png_yes <- all(grepl(x = data, pattern = ".png"))
  image_yes <- jpg_yes | png_yes
  image_yes
}

#' Initializes a Subnetwork based on the Processed Additive Predictor
#' 
#' @param pp list of processed predictor lists from \code{processor}
#' @param deep_top keras layer if the top part of the deep network after orthogonalization
#' is different to the one extracted from the provided network 
#' @param orthog_fun function used for orthogonalization
#' @param split_fun function to split the network to extract head
#' @param shared_layers list defining shared weights within one predictor;
#' each list item is a vector of characters of terms as given in the parameter formula
#' @param param_nr integer number for the distribution parameter
#' @param selectfun_in,selectfun_lay functions defining which subset of pp to
#' take as inputs and layers for this subnetwork; per default the \code{param_nr}'s entry
#' @param gaminputs input tensors for gam terms
#' @param summary_layer keras layer that combines inputs (typically adding or concatenating)
#' @return returns a list of input and output for this additive predictor
#' 
#' @export
#' 
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
  
  # subnetwork builder for torch is still rudimentary. It only initializes the
  # different layers and names them
  # Main difference to the tensorflow approach is that the builder don't has a 
  # input output flow. So the best idea is to maintain different subnetwork 
  # builder for the approaches.
  
  pp_in <- selectfun_in(pp)
  pp_lay <- selectfun_lay(pp)
  
 
  layer_matching <- 1:length(pp_in)
  names(layer_matching) <- layer_matching
  
  
  if(all(sapply(pp_in, function(x) is.null(x$right_from_oz)))){ 
    # if there is no term to orthogonalize
    
      outputs <- lapply(1:length(pp_in),
                        function(i) pp_lay[[layer_matching[i]]]$layer())
      names(outputs) <- sapply(1:length(pp_in),
                               function(i) pp_lay[[layer_matching[i]]]$term)
      return(outputs) 
  
  }
}

#' @title Compile a Deep Distributional Regression Model (Torch)
#'
#'
#' @param list_pred_param list of output(-lists) generated from
#' \code{subnetwork_init}
#' @param weights vector of positive values; optional (default = 1 for all observations)
#' @param optimizer optimizer used. Per default Adam
#' @param model_fun which function to use for model building (default \code{keras_model})
#' @param monitor_metrics Further metrics to monitor
#' @param from_preds_to_output function taking the list_pred_param outputs
#' and transforms it into a single network output
#' @param loss the model's loss function; per default evaluated based on
#' the arguments \code{family} and \code{weights} using \code{from_dist_to_loss}
#' @param additional_penalty a penalty that is added to the negative log-likelihood;
#' must be a function of model$trainable_weights with suitable subsetting
#' @param ... arguments passed to \code{from_preds_to_output}
#' @return a list with input tensors and output tensors that can be passed
#' to, e.g., \code{keras_model}
#'
torch_dr <- function(
    list_pred_param,
    optimizer = torch::optim_adam,
    model_fun = NULL, #nicht mehr nötig bei Torch?
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

#' @title Define Predictor of a Deep Distributional Regression Model
#'
#'
#' @param list_pred_param list of output(-lists) generated from
#' \code{subnetwork_init}
#' @param family see \code{?deepregression}; if NULL, concatenated
#' \code{list_pred_param} entries are returned (after applying mapping if provided)
#' @param output_dim dimension of the output
#' @param mapping a list of integers. The i-th list item defines which element
#' elements of \code{list_pred_param} are used for the i-th parameter.
#' For example, \code{mapping = list(1,2,1:2)} means that \code{list_pred_param[[1]]}
#' is used for the first distribution parameter, \code{list_pred_param[[2]]} for
#' the second distribution parameter and  \code{list_pred_param[[3]]} for both
#' distribution parameters (and then added once to \code{list_pred_param[[1]]} and
#' once to \code{list_pred_param[[2]]})
#' @param from_family_to_distfun function to create a \code{dist_fun} 
#' (see \code{?distfun_to_dist}) from the given character \code{family}
#' @param from_distfun_to_dist function creating a tfp distribution based on the
#' prediction tensors and \code{dist_fun}. See \code{?distfun_to_dist}
#' @param add_layer_shared_pred layer to extend shared layers defined in \code{mapping}
#' @param trafo_list a list of transformation function to convert the scale of the
#' additive predictors to the respective distribution parameter
#' @return a list with input tensors and output tensors that can be passed
#' to, e.g., \code{keras_model}
#'
#' @export
from_preds_to_dist_torch <- function(
    list_pred_param, family = NULL,
    output_dim = 1L,
    mapping = NULL, # not implemented
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

#' @title Function to define output distribution based on dist_fun
#'
#' @param dist_fun a distribution function as defined by \code{make_torch_dist}
#' @param preds tensors with predictions
#' @return a symbolic torch distribution
#' @export
#'
from_distfun_to_dist_torch <- function(dist_fun, preds){
  nn_module(
    
    initialize = function() {
      
      self$distr_parameters <- nn_module_list(
        lapply(preds, function(x) x()))
      names(self$distr_parameters$.__enclos_env__$private$modules_) <- 
        names(preds)
    },
    
    forward = function(dataset_list) {
      distribution_parameters <- lapply(
        1:length(self[[1]]), function(x){
          self[[1]][[x]](dataset_list[[x]])
        })
      dist_fun(distribution_parameters)
    }
  )
}





#' Families for deepregression
#'
#' @param family character vector
#'
#' @details
#' To specify a custom distribution, define the a function as follows
#' \code{
#' function(x) do.call(your_tfd_dist, lapply(1:ncol(x)[[1]],
#'                                     function(i)
#'                                      your_trafo_list_on_inputs[[i]](
#'                                        x[,i,drop=FALSE])))
#' }
#' and pass it to \code{deepregression} via the \code{dist_fun} argument.
#' Currently the following distributions are supported
#' with parameters (and corresponding inverse link function in brackets):
#'
#' \itemize{
#'  \item{"normal": }{normal distribution with location (identity), scale (exp)}
#'  \item{"bernoulli": }{bernoulli distribution with logits (identity)}
#'  \item{"exponential": }{exponential with lambda (exp)}
#'  \item{"gamma": }{gamma with concentration (exp) and rate (exp)}
#'  \item{"poisson": }{poisson with rate (exp)}
#' @param add_const small positive constant to stabilize calculations
#' @param trafo_list list of transformations for each distribution parameter.
#' Per default the transformation listed in details is applied.
#' @param output_dim number of output dimensions of the response (larger 1 for
#' multivariate case) (not implemented yet)
#'
#' @export
#' @rdname dr_families
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

#' Character-torch mapping function
#' 
#' @param family character defining the distribution
#' @return a torch distribution
#' @export
family_to_trochd <- function(family){
  # define dist_fun
  torchd_dist <- switch(family,
                        normal = distr_normal,
                        bernoulli = distr_bernoulli, 
                        gamma = distr_gamma,
                        poisson = distr_poisson)
}

#' Character-to-transformation mapping function
#' 
#' @param family character defining the distribution
#' @param add_const see \code{\link{make_torch_dist}}
#' @return a list of transformation for each distribution parameter
#' @export
#' 
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

#' Function to create (custom) family
#' 
#' @param torch_dist a torch probability distribution
#' @param trafo_list list of transformations h for each parameter 
#' (e.g, \code{exp} for a variance parameter)
#' @param output_dim integer defining the size of the response
#' @return a function that can be used to train a 
#' distribution learning model in torch
#' @export
#' 
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

#' Function to transform a distribution layer output into a loss function
#'
#' @param family see \code{?deepregression}
#' @param weights sample weights
#'
#' @return loss function
from_dist_to_loss_torch <- function(family, weights = NULL){
  
  # define weights to be equal to 1 if not given
  #if(is.null(weights)) weights <- 1
  
  # the negative log-likelihood is given by the negative weighted
  # log probability of the dist
  if(family != "normal") stop("Only normal distribution tested")
  negloglik <- function(input, target) torch_mean(-input$log_prob(target))
  negloglik
  }
  

# Ordering of data is different between torch and tensorflow approach, because
# gaminputs are handled seperate in tensorflow.
# This leads to the fact that prepare_data() can't be used directly
prepare_data_torch <- function(pfc, input_x, target = NULL){
  
  distr_datasets_length <- sapply(
    1:length(pfc),
    function(x) length(pfc[[x]])
    )
  
  sequence_length <- seq_len(sum(distr_datasets_length))
  
  distr_datasets_index <- lapply(distr_datasets_length, function(x){
    index <- sequence_length[seq_len(x)]
    sequence_length <<- sequence_length[-seq_len(x)]
    index
  })
  
  df_list <- lapply(distr_datasets_index, function(x){ input_x[x] })
  
  if(is.null(target)) return(df_list)
  
  get_luz_dataset(df_list = df_list, target = torch_tensor(target))

    
}

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

prepare_input_list_model <- function(input_x, input_y,
                                     object, epochs = 10, batch_size = 32,
                                     validation_split, validation_data = NULL,
                                     callbacks = NULL, verbose, view_metrics, 
                                     early_stopping){
  
  if(object$engine == "torch"){
    input_dataloader <- prepare_data_torch(
      pfc  = object$init_params$parsed_formulas_content,
      input_x = input_x,
      target = input_y)
    
    #no validation
    if(is.null(validation_data) & identical(validation_split, 0)){
      train_dl <- dataloader(input_dataloader, batch_size = batch_size)
      valid_dl <- NULL
    }
    
    if(!is.null(validation_data)){
      train_dl <- dataloader(input_dataloader, batch_size = batch_size)
      
      validation_dataloader <- prepare_data_torch(
        pfc  = object$init_params$parsed_formulas_content,
        input_x = validation_data[[1]],
        target = validation_data[[2]])
      
      valid_dl <- dataloader(validation_dataloader, batch_size = batch_size)
    }
    
    if(!identical(validation_split, 0) & !is.null(validation_split)){
      train_ids <- sample(1:length(input_y), 
                          size = ceiling((1-validation_split) * length(input_y)))
      valid_ids <- sample(setdiff(1:length(input_y), train_ids),
                          size = validation_split * length(input_y))
      
      train_ds <- dataset_subset(input_dataloader, indices = train_ids)
      valid_ds <- dataset_subset(input_dataloader, indices = valid_ids)
      
      if(any(unlist(lapply(input_x, check_data_for_image)))) 
        cat(sprintf("Found %s validated image filenames \n", length(train_ids)))
      train_dl <- dataloader(train_ds, batch_size = batch_size)
      if(any(unlist(lapply(input_x, check_data_for_image))))
        cat(sprintf("Found %s validated image filenames \n", length(valid_ids)))
      valid_dl <- dataloader(valid_ds, batch_size = batch_size)
    }
    
    if(!is.null(valid_dl)) valid_data <- valid_dl
    if(is.null(valid_dl)) valid_data <- NULL
    
    input_list_model <- list(
      object = object$model,
      epochs = epochs,
      data = train_dl,
      valid_data = valid_data,
      callbacks = callbacks,
      verbose = verbose)
    
    #if(view_metrics) stop("Only possible in tf approach. Use luz callbacks instead")
    #if(verbose) stop("Only possible in tf approach. Use luz callbacks instead")
    
    return(c(input_list_model))
  }
  
  if(object$engine == 'tf'){
    input_list_model <-
      list(object = object$model,
           epochs = epochs,
           batch_size = batch_size,
           validation_split = validation_split,
           validation_data = validation_data,
           callbacks = callbacks,
           verbose = verbose,
           view_metrics = ifelse(view_metrics, getOption("keras.view_metrics", default = "auto"), FALSE)
      )
    
      input_list_model <- c(input_list_model,
                            list(x = input_x, y = input_y)
                            )
    
      input_list_model}
    }



  