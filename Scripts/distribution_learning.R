library(deepregression)
library(torch)
library(luz)

torch_manual_seed(42)
x <- torch_arange(0, 1, step = 0.001)$view(c(1001,1))
#y <- 2*x + 1 + torch_randn(100, 1) # mean von y
intercept <- 1
beta1 <- 2

n <- distr_normal(loc = beta1*x, scale = torch_exp(beta1*x))
y <- n$sample(1)
plot(as.numeric(x), as.numeric(y))

GaussianLinear <- nn_module(
  initialize = function() {
    # this linear predictor will estimate the mean of the normal distribution
    self$linear <- nn_linear(1, 1, bias = F)
    # this parameter will hold the estimate of the variability
    self$scale <- nn_linear(1, 1, bias = F)
  },
  forward = function(x) {
    # we estimate the mean
    loc <- self$linear(x[[1]])
    scale <-  self$scale(x[[2]])
    # return a normal distribution
    #distr_normal(loc, self$scale)
    distr_normal(loc, torch_exp(scale))
    
  }
)

model <- GaussianLinear()
model$parameters
x_list <- list(x, x)
opt <- optim_adam(model$parameters)

for (i in 1:2000) {
  opt$zero_grad()
  d <- model(x_list)
  loss <- torch_mean(-d$log_prob(y))
  loss$backward()
  opt$step()
  if (i %% 10 == 0)
    cat("iter: ", i, " loss: ", loss$item(), "\n")
}
gamlss::gamlss(formula = as.numeric(y)~ -1+as.numeric(x),
       sigma.formula = ~ -1 + as.numeric(x), family = "NO")
model$parameters
# true ist 2 also passt es

################################################################################
################################################################################
###################### Deepregression example with torch ######################$
################################################################################
################################################################################

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

#debugonce(deepregression)
mod <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y,orthog_options = orthog_options,
  list_of_deep_models = list(deep_model = deep_model), engine = "tf"
)

if(!is.null(mod)){
  # train for more than 10 epochs to get a better model
  mod %>% fit(epochs = 100, early_stopping = TRUE)
  mod %>% coef()
  mod %>% plot()
  mod %>% fitted()
}


source('Scripts/deepregression_functions.R')
# So muss deep_model bei torch ansatz angegeben werden, da kein Input vorhanden ist
deep_model <- nn_sequential(
  nn_linear(in_features = 3, out_features = 32, bias = F),
  nn_relu(),
  nn_dropout(p = 0.2),
  nn_linear(in_features = 32, out_features = 8),
  nn_relu(),
  nn_linear(in_features = 8, out_features = 1)
)

# oder so 

deep_model <- function() nn_sequential(
  nn_linear(in_features = 3, out_features = 32, bias = F),
  nn_relu(),
  nn_dropout(p = 0.2),
  nn_linear(in_features = 32, out_features = 8),
  nn_relu(),
  nn_linear(in_features = 8, out_features = 1)
)



# Soll subnetwork_init nachstellen
# Gebe liste von layern pro Parameter aus
# brauche keine Inputs, also nur Layer initialisieren
mu_submodules <- list(deep_model(),
                      torch_layer_dense(units = 1),
                      layer_spline_torch(units = 9,
                                      P = torch_tensor(
                                        matrix(rep(0, 9*9), ncol = 9))),
                      torch_layer_dense(units = 1))
sigma_submodules <- list(torch_layer_dense(units = 1))

submodules <- list(mu_submodules, sigma_submodules)

neural_net <- lapply(submodules, torch_model)
neural_net[[1]]()$forward
neural_net[[2]]()$forward
neural_net[[1]]()$parameters
neural_net[[2]]()$parameters

# hi3r überall data_trafo() nutzen
deep_model_data <- mod$init_params$parsed_formulas_contents$loc[[1]]$data_trafo()
gam_data <- mod$init_params$parsed_formulas_contents$loc[[2]]$data_trafo()
lin_data <- mod$init_params$parsed_formulas_contents$loc[[3]]$data_trafo()
int_data <- mod$init_params$parsed_formulas_contents$loc[[4]]$data_trafo()


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
# DS pro Parameter unterschiedlich sein können
get_luz_dataset <- dataset(
  "deepregression_luz_dataset",
  
  initialize = function(mu_df_list, sigma_df_list, target) {
    self$df_list <- list(mu_df_list, sigma_df_list)
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

luz_dataset <- get_luz_dataset(mu_df_list = mu_inputs_list,
                               sigma_df_list = sigma_inputs_list,
                               target  = torch_tensor(y))
luz_dataset$.getitem(1)

train_dl <- dataloader(luz_dataset, batch_size = 32, shuffle = F)

# entweder loss extern oder intern angeben
# scheinbar eher innerhalb angeben
# muss sonst mehrmals aktiviert werden (Siehe unten)
# 
distr_loss <- function(){
  nn_module(
    loss = function(input, target){
      torch_mean(-input$log_prob(target))
    })
}
loss_test <- distr_loss()
loss_test()$loss

distribution_learning <- function(neural_net_list, family){
  nn_module(
    initialize = function() {
    
      self$distr_parameters <- nn_module_list(
        lapply(neural_net_list, function(x) x()))

      # this linear predictor will estimate the mean of the normal distribution
      #self$linear <- nn_linear(1, 1, bias = F)
      # this parameter will hold the estimate of the variability
      #self$scale <- nn_linear(1, 1, bias = F)
    },
    
    forward = function(dataset_list) {
      # we estimate the mean
      distribution_parameters <- lapply(
        1:length(self$distr_parameters), function(x){
        self$distr_parameters[[x]](dataset_list[[x]])
        })
      
      # check which distributions are already implemented
      #distr_normal(distribution_parameters[[1]], distribution_parameters[[2]])
      do.call(family, distribution_parameters)
      #distr_normal(preds_loc, torch_exp(preds_sigma))
    },
    
  loss = function(input, target){
    torch_mean(-input$log_prob(target))
  })
}
distr_learning <- distribution_learning(neural_net, family = distr_normal)
distr_learning()

pre_fitted <- distr_learning %>% 
  setup(
    optimizer = optim_adam
  ) %>%
  set_opt_hparams(lr = 0.1) 
fitted <- pre_fitted %>% fit(
  data = train_dl, epochs = 10)

plot(mod %>% fitted(),
     as.array(neural_net[[1]]()$forward(mu_inputs_list)))

