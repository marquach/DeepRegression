library(torch)
library(luz)
library(torchvision)
devtools::load_all("deepregression-main/")

orthog_options = orthog_control(orthogonalize = F)

# load  and prepare data
airbnb <- readRDS("/Users/marquach/Desktop/R_Projects/semi-structured_distributional_regression/application/airbnb/munich_clean.RDS")
airbnb$days_since_last_review <- as.numeric(
  difftime(airbnb$date, airbnb$last_review)
)
y = log(airbnb$price)

ens_mod_tf <- deepregression(
 y = y,
 data = airbnb,
 list_of_formulas = list(
   location = ~ 1 + s(longitude),
   scale = ~ 1 + s(longitude)
   ),   orthog_options = orthog_options)

ens_mod_torch <- deepregression(
  y = y,
  data = airbnb,
  list_of_formulas = list(
    location = ~ 1 + s(longitude),
    scale = ~ 1 + s(longitude)
  ),
  engine = "torch",
  orthog_options = orthog_options)

ens_tf <- ensemble(ens_mod_tf, n_ensemble = 3, epochs = 250,
                   early_stopping = F,
                    validation_split = 0.2)
ens_torch <- ensemble(ens_mod_torch, n_ensemble = 3, epochs = 100,
                      early_stopping = F,
                   validation_split = 0.2, verbose = F)

ensemble_distr_tf <- get_ensemble_distribution(ens_tf)
ensemble_distr_tf$sample()
ensemble_distr_tf$mean()

ensemble_distr_torch <- get_ensemble_distribution(ens_torch)
ensemble_distr_torch$sample()
# not implemented
ensemble_distr_torch$mean()

coef(ens_torch)
coef(ens_tf)
str(fitted(ens_torch),1)
str(fitted(ens_tf),1)




# different distributions

set.seed(42)
n <- 100
x <- sample(x = log(85:125), size = n, replace = T)
mu <- -0.5 + 2*x
y <- rpois(n = n, lambda = exp(mu))
data_poisson <- data.frame(x)

model_poisson_torch <- deepregression(y = y,
                                      list_of_formulas = list( rate = ~1 + x),
                                      data = data_poisson,
                                      family = "poisson",
                                      engine = "torch",
                                      orthog_options = orthog_options,
                                      subnetwork_builder = subnetwork_init_torch,
                                      model_builder = torch_dr)
model_poisson_torch$model <- model_poisson_torch$model  %>%
  set_opt_hparams(lr = 0.1)
ens_torch_poisson <- ensemble(model_poisson_torch, n_ensemble = 3, epochs = 100,
                      early_stopping = F,
                      validation_split = 0.2, verbose = F)
ensemble_distr_torch_poisson <- get_ensemble_distribution(ens_torch_poisson)
ensemble_distr_torch_poisson$sample()
coef(ens_torch_poisson)
str(fitted(ens_torch_poisson),1)


# Gamma (no natural link)
set.seed(42)
n <- 100
x <- runif(n, -1, 1)
a <- 0.5
b <- 1.2
y_true <- exp(a + b * x)
shape <- exp(0)
scale_distr = y_true/shape
y <- rgamma(n, rate = scale_distr, shape = shape)
data_gamma <- data.frame(x = x)

model_gamma_torch <- deepregression(y = y,
                                    list_of_formulas = list(
                                      concentration = ~ 1 ,
                                      rate = ~ 1+ x),
                                    data = data_gamma,
                                    family = "gamma",
                                    engine = "torch",
                                    orthog_options = orthog_options)

model_gamma_torch$model <- model_gamma_torch$model  %>%
  set_opt_hparams(lr = 0.1)
ens_torch_gamma <- ensemble(model_gamma_torch, n_ensemble = 3, epochs = 10,
                              early_stopping = F,
                              validation_split = 0.2, verbose = F)
ensemble_distr_torch_gamma <- get_ensemble_distribution(ens_torch_gamma)
ensemble_distr_torch_gamma$sample()
coef(ens_torch_gamma, which_param = 1)
coef(ens_torch_gamma, which_param = 2)
str(fitted(ens_torch_gamma),1)


# bernoulli
set.seed(42)
n <- 1000
x <- runif(n, -1, 1)
# plogis == inverse logit
y <- rbinom(n = length(x), size = 1, prob = plogis(1 + 2*x))
data_bern <- data.frame(x)
glm(y~x, family = binomial)

model_bernoulli_torch <- deepregression(y = y,
                                        list_of_formulas = list( 
                                          logit = ~ 1 + x),
                                        data = data_bern,
                                        family = "bernoulli",
                                        engine = "torch",
                                        orthog_options = orthog_options)
ens_torch_bernoulli <- ensemble(model_bernoulli_torch, n_ensemble = 3, epochs = 200,
                              early_stopping = F,
                              validation_split = 0.2, verbose = T)
ensemble_distr_torch_bernoulli <- get_ensemble_distribution(ens_torch_bernoulli)
ensemble_distr_torch_bernoulli$sample()
coef(ens_torch_bernoulli)
str(fitted(ens_torch_bernoulli),1)
