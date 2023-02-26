# Verständnis aus distribution_learning
library(deepregression)
library(torch)
library(luz)
devtools::load_all("deepregression-main/")

set.seed(42)
n <- 1000

data = data.frame(matrix(rnorm(4*n), c(n,4)))
colnames(data) <- c("x1","x2","x3","xa")
y <- rnorm(n) + data$xa^2 + data$x1

deep_model <- function(x) x %>%
  layer_dense(units = 32, activation = "relu", use_bias = FALSE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 8, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")

orthog_options = orthog_control(orthogonalize = F)

formula <- ~ 1 + deep_model(x1,x2,x3) + s(xa) + x1

debugonce(deepregression)
mod <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y,orthog_options = orthog_options,
  list_of_deep_models = list(deep_model = deep_model), engine = "tf"
)
deep_model_data <- mod$init_params$parsed_formulas_contents$loc[[1]]$data_trafo()
gam_data <- mod$init_params$parsed_formulas_contents$loc[[2]]$data_trafo()
lin_data <- mod$init_params$parsed_formulas_contents$loc[[3]]$data_trafo()
int_data <- mod$init_params$parsed_formulas_contents$loc[[4]]$data_trafo()
deep_model_data <- torch_tensor(as.matrix(as.data.frame(deep_model_data)))
lin_data <- torch_tensor(lin_data)
gam_data <- torch_tensor(gam_data)
int_data <- torch_tensor(int_data)


######
# Soll subnetwork_init nachstellen
# Gebe liste von layern pro Parameter aus
# brauche keine Inputs, also nur Layer initialisieren
deep_model <- function() nn_sequential(
  nn_linear(in_features = 3, out_features = 32, bias = F),
  nn_relu(),
  nn_dropout(p = 0.2),
  nn_linear(in_features = 32, out_features = 8),
  nn_relu(),
  nn_linear(in_features = 8, out_features = 1)
)
mu_submodules <- list(deep_model(),
                      layer_dense_torch(units = 1),
                      layer_spline_torch(units = 9,
                                         P = 
                                           matrix(rep(0, 9*9), ncol = 9)),
                      layer_dense_torch(units = 1))
sigma_submodules <- list(layer_dense_torch(units = 1))

submodules <- list(mu_submodules, sigma_submodules)

input1 <- torch_tensor(as.matrix(data[,1:3]))
input2 <- torch_tensor(lin_data)
input3 <- torch_tensor(gam_data)
input4 <- torch_tensor(int_data)

mu_inputs_list <- list(input1,
                       input2,
                       input3,input4)
sigma_inputs_list <- list(input4)
test_list <- list(mu_inputs_list, sigma_inputs_list)

# Erstelle Datensatz für luz setup 
# Bilde pro Verteilungsparameter eine Liste in der Daten enthalten sind, da 
# DS pro Parameter unterschiedlich sein können und
# erstelle Liste mit Listen als Element.


luz_dataset <- get_luz_dataset(df_list = test_list,
                               target  = torch_tensor(y))
luz_dataset$.getitem(1)

train_dl <- dataloader(luz_dataset, batch_size = 32, shuffle = F)

# entweder loss extern oder intern angeben
# scheinbar eher innerhalb angeben
# muss sonst mehrmals aktiviert werden (Siehe unten)
# 
dis_loss = function(input, target){
  torch_mean(-input$log_prob(target))
}

outputs <- lapply(submodules, model_torch)

distr_learning <- distribution_learning(outputs, family = "normal")
pre_fitted <- distr_learning %>% 
  luz::setup(
    optimizer = optim_adam, 
    loss = dis_loss)
pre_fitted <- pre_fitted %>%
  set_opt_hparams(lr = 0.1) 
fit_done <- pre_fitted %>% fit(
  data = train_dl, epochs = 100)

plot(mod %>% fitted(),
     as.array(distr_learning()[[1]][[1]]$forward(mu_inputs_list)))

























# from_preds_to_dist nachstellen
# Input sind hier:
#- list_pred_param,
#- family = NULL,
#- output_dim = 1L,
#- mapping = NULL,
#- from_family_to_distfun = make_tfd_dist,
#- from_distfun_to_dist = distfun_to_dist,
#- add_layer_shared_pred = function(x, units) layer_dense(x, units = units,
#                                                       use_bias = FALSE),
#- trafo_list = NULL

# Alles was schon in meinem distribution_learning enthalten ist
# Sollte man ja dann alles da rein packen können und später nur ganz simple
# Model Fun benutzen


from_preds_to_dist_torch <- function(
    list_pred_param, family = NULL,
    output_dim = 1L,
    mapping = NULL,
    from_family_to_distfun = make_torch_dist,
    from_distfun_to_dist = distfun_to_dist_torch,
    #add_layer_shared_pred = function(x, units) layer_dense_torch(x, units = units,
    #                                                       use_bias = FALSE),
    trafo_list = NULL){
  
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
  
  preds <- test
  
  # generate output
  
  out <- from_distfun_to_dist_torch(dist_fun, preds)
  
}
    



# Input sind Netzwerke
# Hier drinnen Netzwerke verbinden über model torch
# und vll noch Trafo List nutzen

distribution_learning <- function(neural_net_list, family, output_dim = 1L){
  nn_module(
    "distribution_learning_module",
    initialize = function() {
      
      self$distr_parameters <- nn_module_list(
        lapply(neural_net_list, function(x) x()))
    },
    
    forward = function(dataset_list) {
      distribution_parameters <- lapply(
        1:length(self$distr_parameters), function(x){
          self$distr_parameters[[x]](dataset_list[[x]])
        })
      
      dist_fun <- make_torch_dist(family, output_dim = output_dim)
      do.call(dist_fun, list(distribution_parameters))
    },
    
    #loss = function(input, target){
    #  torch_mean(-input$log_prob(target))
    #}
  )
}


######