library(luz)
library(torch)

#get data
set.seed(42)
n <- 1000

data = data.frame(matrix(rnorm(4*n), c(n,4)))
colnames(data) <- c("x1","x2","x3","xa")
y <- rnorm(n) + data$xa^2 + data$x1

# Test

gam_model <- mgcv::gam(y ~ 1 + x1 + s(xa), data = data, family = gaussian)
gam_model$coefficients

#param_nr adden


#model_builder aus deepregression
# Input
#list_pred_param,
#weights = NULL,
#optimizer = tf$keras$optimizers$Adam(),
#model_fun = keras_model,
#monitor_metrics = list(),
#from_preds_to_output = from_preds_to_dist,
#loss = from_dist_to_loss(family = list(...)$family,
#                         weights = weights),
#additional_penalty = NULL,
#...
## => später bauen

# Nimmt als input eine liste von submodulen (pro Verteilungparameter eine Liste)

torch_model <-  function(submodules_list){
  nn_module(
  classname = "torch_model",
  initialize = function() {
    self$subnetworks <- nn_module_list(submodules_list)
  },
  
  forward = function(dataset_list) {
    # Problem ist das setup nur eine Liste mit zwei Einträge akzeptiert
    # Daten und Target. 
    # Kann auch nicht umgegangen werden indem erster Eintrag ne Liste ist
    # Liegt daran, dass intern dataloader genutzt wird und der erwartet
    # natürlich einen einzelnen Datensatz und keine Liste mit mehreren Datensätzen
    # Input ist ein Tensor
    # index <- list(1:9,10,11) # das muss natürlich weg, aber noch keine gute 
    # Idee. Sozusgen Spalte 1:9 sind Daten für Spline, usw.
    # Könnte Indexe angeben wo daten der Module liegen.
    # Setz aber voraus, dass alle Tensoren  z.b. 
    # die gleiche erste Dimension haben. Daher eher schlecht
    # Das sollte zu Problemen bei Bilder oder 
    # anderen nicht tabellarischen Daten führen
    
    # Neue Idee: direkt einen dataloader geben
    # Mit Dataloader möglich mit Listen zu arbeiten. 
    # Man muss nur einen Dataloader dafür bauen. Hab ich gemacht und
    # sollte funktionieren. Noch sehr simple muss vll verändert werden
    
    subnetworks <- lapply(1:length(self$subnetworks), function(x){
        self$subnetworks[[x]](dataset_list[[x]])
    })
    Reduce(f = "+", x = subnetworks)}
  )}


get_luz_dataset <- dataset(
  "mydataset",
  
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
  
  # Dataloader müsste dann m.M.n. in deepregression erstellt werden,
  # da außerhalb nicht schön wäre bzw. zu sehr von keras abweicht.
  # Input wird ja eh schon in Form von data gegeben, also sollte das passen
  
  .length = function() {
    length(self$target)
  }
  
)

spline_layer <- nn_module(
  classname = "spline_module", 
  
  initialize = function() {
    self$spline <- nn_linear(in_features = 9, out_features = 1, bias = F)
      
      #layer_spline(units = 9, #mod_true_preproc$loc[[1]]$input_dim
                                #name = "spline_layer")
  },
  forward = function(x) {
    self$spline(x)}
)

intercept_layer <- nn_module(
  classname = "intercept_module",
  initialize = function() {
    self$intercept <- nn_linear(1, 1, F) # Bias
  },
  forward = function(x) {
    self$intercept(x)
  }
)

x1_layer <- nn_module(
  classname = "x1_module",
  initialize = function() {
    self$weightx1 <- nn_linear(in_features = 1, out_features = 1, bias = F)
  },
  forward = function(x) {
    self$weightx1(x)
  }
)

# subnetwork_init sollte bei torch nur die einzelnen Layer als Listeneinträge 
# ausgeben. Hier wird kein Input mehr ausgegeben, da bei Torch nicht notwendig.
# Also eine Liste von Layern pro Verteilungsparameter
# Hier nur ein Beispiel für loc gemacht:
submodules <- list(spline_layer(),
                   intercept_layer(),
                   x1_layer()
                   )
# submodules ist output von subnetwork_init

# torch_model ist wie keras_model
neural_net <- torch_model(submodules)
neural_net()$forward # Schaut gut aus
neural_net()$parameters # Schaut gut aus

# hi3r dann überall data_trafo() nutzen
input1 <- torch_tensor(model.matrix(gam_model)[,-c(1,2)]) 
input2 <- torch_tensor(rep(1, 1000))$view(c(1000, 1))
input3 <- torch_tensor(data$x1)$view(c(1000, 1))
inputs_list <- list(input1, input2, input3)

luz_dataset <- get_luz_dataset(inputs_list, target  = torch_tensor(y))

luz_dataset$.getitem(1) # Schaut gut aus, da Target [[2]] ist. 
# Das setzt fit von luz voraus.
train_dl <- dataloader(luz_dataset, batch_size = 1000, shuffle = F)
pre_fitted <- neural_net %>% 
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
  ) %>% 
  set_opt_hparams(lr = 0.1, weight_decay=0.01) 

fitted <- pre_fitted %>% fit(
  data = train_dl, epochs = 200)

plot(gam_model$fitted.values, fitted %>% predict(train_dl),
     xlab = "GAM", ylab = "Neural Net")
abline(a = 0, b = 1, lty = 2)  

plot(gam_model)
points(data$xa,
       model.matrix(gam_model)[,-c(1,2)]%*%t(as.matrix(neural_net()$parameters[[1]])),
       col="red")
