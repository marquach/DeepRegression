################################################################################
################################################################################
###################### Deepregression example with torch ######################$
################################################################################
################################################################################
library(deepregression)
library(torch)
library(luz)
devtools::load_all("deepregression-main/")

set.seed(42)
n <- 1000

data = data.frame(matrix(rnorm(4*n), c(n,4)))
colnames(data) <- c("x1","x2","x3","xa")
y <- rnorm(n) + data$xa^2 + data$x1

deep_model_tf <- function(x) x %>%
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
  list_of_deep_models = list(deep_model = deep_model_tf), engine = "tf"
)

if(!is.null(mod)){
  # train for more than 10 epochs to get a better model
  mod %>% fit(epochs = 100, early_stopping = TRUE)
}

source('Scripts/deepregression_functions.R')
# Für subnetwork_init_torch und sollen später in passendes Skript verschoben werden
# So muss deep_model bei torch ansatz angegeben werden, da kein Input vorhanden ist
deep_model_torch <- function() nn_sequential(
  nn_linear(in_features = 3, out_features = 32, bias = F),
  nn_relu(),
  nn_dropout(p = 0.2),
  nn_linear(in_features = 32, out_features = 8),
  nn_relu(),
  nn_linear(in_features = 8, out_features = 1)
)

#debugonce(deepregression)
mod_torch <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1), 
  list_of_deep_models = list(deep_model = deep_model_torch),
  data = data, y = y, orthog_options = orthog_options, return_prepoc = F, 
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  engine = "torch")

mod_torch %>% fit(epochs = 100, early_stopping = T)

mod %>% plot()
points(data$xa,
       torch_matmul(mod_torch$init_params$parsed_formulas_contents$loc[[2]]$data_trafo(),
                 mod_torch$model()[[1]][[1]]$parameters[6]$sub$t()), col="red")

deep_model_data <- mod$init_params$parsed_formulas_contents$loc[[1]]$data_trafo()
gam_data <- mod$init_params$parsed_formulas_contents$loc[[2]]$data_trafo()
lin_data <- mod$init_params$parsed_formulas_contents$loc[[3]]$data_trafo()
int_data <- mod$init_params$parsed_formulas_contents$loc[[4]]$data_trafo()
deep_model_data <- torch_tensor(as.matrix(as.data.frame(deep_model_data)))
lin_data <- torch_tensor(lin_data)
gam_data <- torch_tensor(gam_data)
int_data <- torch_tensor(int_data)
mu_inputs_list <- list(deep_model_data,
                       gam_data,
                       lin_data,
                       int_data)

plot(mod %>% fitted(),
     mod_torch$model()[[1]][[1]]$forward(mu_inputs_list))
cor(mod %>% fitted(),
     as.array(mod_torch$model()[[1]][[1]]$forward(mu_inputs_list)))
