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
}

source('Scripts/deepregression_functions.R')
# Für subnetwork_init_torch und sollen später in passendes Skript verschoben werden
# So muss deep_model bei torch ansatz angegeben werden, da kein Input vorhanden ist
deep_model <- function() nn_sequential(
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
  list_of_deep_models = list(deep_model = deep_model),
  data = data, y = y, orthog_options = orthog_options, return_prepoc = F, 
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  engine = "torch")


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
sigma_inputs_list <- list(int_data)
test_list <- list(mu_inputs_list, sigma_inputs_list)

luz_dataset <- get_luz_dataset(df_list = test_list,
                               target  = torch_tensor(y))
train_dl <- dataloader(luz_dataset, batch_size = 32, shuffle = F)

mod_torch$model <- mod_torch$model %>%
  set_opt_hparams(lr = 0.1)

fit_done <- mod_torch$model %>% luz::fit(
  data = train_dl, epochs = 50)

plot(mod %>% fitted(),
     mod_torch$model()[[1]][[1]]$forward(mu_inputs_list))

mod %>% plot()
points(data$xa,
       as.array(gam_data)%*%
         t(as.array(mod_torch$model()[[1]][[1]]$parameters[6]$sub)), col="red")



#now with validation
train_ids <- sample(1:dim(data)[1], size = 0.6 * dim(data)[1])
valid_ids <- sample(setdiff(1:dim(data)[1], train_ids), size = 0.2 * dim(data)[1])

train_ds <- dataset_subset(luz_dataset, indices = train_ids)
valid_ds <- dataset_subset(luz_dataset, indices = valid_ids)

train_dl <- dataloader(train_ds, batch_size = 29)
valid_dl <- dataloader(valid_ds, batch_size = 29)

mod_torch <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1), 
  list_of_deep_models = list(deep_model = deep_model),
  data = data, y = y, orthog_options = orthog_options, return_prepoc = F, 
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  engine = "torch")


mod_torch$model <- mod_torch$model %>%
  set_opt_hparams(lr = 0.1)

fit_done <- mod_torch$model %>% 
  luz::fit(train_dl, epochs = 10, valid_data = valid_dl)

plot(mod %>% fitted(),
     mod_torch$model()[[1]][[1]]$forward(mu_inputs_list))
