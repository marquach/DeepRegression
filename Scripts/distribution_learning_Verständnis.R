# Verständnis aus distribution_learning

######
# Soll subnetwork_init nachstellen
# Gebe liste von layern pro Parameter aus
# brauche keine Inputs, also nur Layer initialisieren
mu_submodules <- list(deep_model(),
                      torch_layer_dense(units = 1),
                      layer_spline_torch(units = 9,
                                         P = 
                                           matrix(rep(0, 9*9), ncol = 9)),
                      torch_layer_dense(units = 1))
sigma_submodules <- list(torch_layer_dense(units = 1))

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
outputs <- lapply(submodules, torch_model)

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
######