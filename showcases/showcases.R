library(deepregression)
library(torch)
library(luz)
devtools::load_all("deepregression-main/")
source('Scripts/deepregression_functions.R')

# Up to this moment (02.27.2023), the torch implementation is only able to handle
# deepregression models which are build without the orthogonalizion.
orthog_options = orthog_control(orthogonalize = F)
# showcase from the deepregression intro presentation

set.seed(42)
n <- 1000
data = data.frame(matrix(rnorm(4*n), c(n,4)))
colnames(data) <- c("x1","x2","x3","xa")

formula <- ~ 1 + deep_model(x1,x2,x3) + s(xa) + x1

deep_model_tf <- function(x) x %>%
  layer_dense(units = 32, activation = "relu", use_bias = FALSE) %>%
  layer_dense(units = 1, activation = "linear")

deep_model_torch <- function() nn_sequential(
  nn_linear(in_features = 3, out_features = 32, bias = F),
  nn_relu(),
  nn_linear(in_features = 32, out_features = 1)
)

y <- rnorm(n) + data$xa^2 + data$x1

mod_tf <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y,orthog_options = orthog_options,
  list_of_deep_models = list(deep_model = deep_model_tf), engine = "tf"
)

mod_torch <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y, orthog_options = orthog_options,
  list_of_deep_models = list(deep_model = deep_model_torch),
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  engine = "torch"
)

mod_tf %>% fit(epochs = 100, early_stopping = T)
mod_tf %>% coef()

# The fit function for the torch approach is not implemented. 
# One needs to prepare a dataloader.

deep_model_data <- mod_tf$init_params$parsed_formulas_contents$loc[[1]]$data_trafo()
gam_data <- mod_tf$init_params$parsed_formulas_contents$loc[[2]]$data_trafo()
lin_data <- mod_tf$init_params$parsed_formulas_contents$loc[[3]]$data_trafo()
int_data <- mod_tf$init_params$parsed_formulas_contents$loc[[4]]$data_trafo()
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

mod_torch$model %>% 
  luz::fit(train_dl, epochs = 100)
# loss pretty high at beginning (initialization bad?)
mod_torch$model()[[1]]
cbind(
  unlist(mod_tf %>% coef()),
  unlist(lapply(mod_torch$model()$parameters[4:6], function(x) t(as.array(x)))))
# not that equal
plot(mod_tf %>% fitted(),
     mod_torch$model()[[1]][[1]]$forward(mu_inputs_list))
cor(mod_tf %>% fitted(),
     as.array(mod_torch$model()[[1]][[1]]$forward(mu_inputs_list)))
# coefs are not that equal, but plot shows they are almost equal and
# also correlation close to 1









