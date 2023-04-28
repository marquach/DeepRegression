library(torch)
library(luz)
library(torchvision)
devtools::load_all("deepregression-main/")
source('scripts/deepregression_functions.R')

set.seed(42)
x <- seq(0, 1, 0.001)
y <- rpois(n = length(x), lambda = exp(2*x))
orthog_options = orthog_control(orthogonalize = F)

plot(density(y))

glm(formula = y~x, family = poisson)

data_poisson <- data.frame(x)

model_poisson_tf <- deepregression(y = y,
                                   list_of_formulas = list( rate = ~1 + x),
               data = data_poisson,
               family = "poisson")
model_poisson_tf %>% fit(epochs = 100, validation_split = 0.2)

model_poisson_torch <- deepregression(y = y,
                                      list_of_formulas = list( rate = ~1 + x),
                                   data = data_poisson,
                                   family = "poisson",
                                   engine = "torch",
                                   orthog_options = orthog_options,
                                   subnetwork_builder = subnetwork_init_torch,
                                   model_builder = torch_dr)
model_poisson_torch %>% fit(epochs = 100, validation_split = 0.2)
model_poisson_torch %>% coef()

set.seed(42)
n <- 10000
x <- seq(0, 1, length.out = n)
y <- rgamma(n, shape = 1, rate = 0.1)
# beta = rate
# alpha = shape (concentration)

glm(y~1,family=Gamma)

model_gamma_tf <- deepregression(y = y,
                                    list_of_formulas = list( rate = ~1,
                                                             shape = ~1),
                                    data = data_poisson,
                                    family = "gamma",
                                    engine = "tf",
                                    orthog_options = orthog_options)
model_gamma_tf %>% fit(epochs = 100, validation_split = 0.2)
model_gamma_tf %>% coef(which_par = 1)
model_gamma_tf %>% coef(which_par = 2)

model_gamma_torch <- deepregression(y = y,
                                      list_of_formulas = list( rate = ~1,
                                                               shape = ~1),
                                      data = data_poisson,
                                      family = "gamma",
                                      engine = "torch",
                                      orthog_options = orthog_options,
                                      subnetwork_builder = subnetwork_init_torch,
                                      model_builder = torch_dr)
model_gamma_torch %>% fit(epochs = 100, validation_split = 0.2)
model_gamma_torch %>% coef()


set.seed(42)
x <- seq(0, 1, 0.0001)
y <- rbinom(n = length(x), size = 1, prob = plogis(1 + 2*x))
glm(y~1 + x, family = "binomial")
data_bern <- data.frame(x)

model_binomial_tf <- deepregression(y = y,
                                       list_of_formulas = list( logit = ~ 1 + x),
                                       data = data_bern,
                                       family = "bernoulli",
                                       engine = "tf",
                                       orthog_options = orthog_options)
model_binomial_tf %>% fit(epochs = 200, validation_split = 0.2)
model_binomial_tf %>% coef()

model_binomial_torch <- deepregression(y = y,
                                      list_of_formulas = list( logit = ~ 1 + x),
                                      data = data_bern,
                                      family = "bernoulli",
                                      engine = "torch",
                                      orthog_options = orthog_options,
                                      subnetwork_builder = subnetwork_init_torch,
                                      model_builder = torch_dr)
model_binomial_torch <- model_binomial_torch %>% set_opt_hparams(lr = 0.1)
model_binomial_torch %>% fit(epochs = 30, validation_split = 0.2)
model_binomial_torch %>% coef()
