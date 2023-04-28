library(torch)
library(luz)
library(torchvision)
devtools::load_all("deepregression-main/")
source('scripts/deepregression_functions.R')

orthog_options = orthog_control(orthogonalize = F)

# load  and prepare data
airbnb <- readRDS("/Users/marquach/Desktop/R_Projects/semi-structured_distributional_regression/application/airbnb/munich_clean.RDS")
airbnb$days_since_last_review <- as.numeric(
  difftime(airbnb$date, airbnb$last_review)
)
y = log(airbnb$price)

mod_tf <- deepregression(
 y = y,
 data = airbnb,
 list_of_formulas = list(
   location = ~ 1 + s(longitude),
   scale = ~ 1 + s(longitude)
   )
  )

mod_torch <- deepregression(
  y = y,
  data = airbnb,
  list_of_formulas = list(
    location = ~ 1 + s(longitude),
    scale = ~ 1 + s(longitude)
  ),
  engine = "torch",
  orthog_options = orthog_options,
  subnetwork_builder = subnetwork_init_torch,
  model_builder = torch_dr)

ens_tf <- ensemble(mod_tf, n_ensemble = 2, epochs = 2, early_stopping = TRUE,
                    validation_split = 0.2)
ens_torch <- ensemble(mod_torch, n_ensemble = 2, epochs = 1, early_stopping = TRUE,
                   validation_split = 0.2, verbose = T)
ens_tf
ens_torch

debugonce(get_ensemble_distribution)
ensemble_distr_tf <- get_ensemble_distribution(ens_tf, data = airbnb)
ensemble_distr_tf$sample()

debugonce(get_ensemble_distribution)
ensemble_distr_torch <- get_ensemble_distribution(ens_torch)
ensemble_distr_torch$sample()


