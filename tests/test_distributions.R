library(torch)
library(luz)
library(torchvision)
library(gamlss)
devtools::load_all("deepregression-main/")

orthog_options = orthog_control(orthogonalize = F)

set.seed(42)
n <- 1000
x <- sample(x = log(85:125), size = n, replace = T)
mu <- -0.5 + 2*x
y <- rpois(n = n, lambda = exp(mu))
data_poisson <- data.frame(x)

model_poisson_tf <- deepregression(y = y*1.0,
                                   list_of_formulas = list( rate = ~ 1 + x),
                                   data = data_poisson,
                                   orthog_options = orthog_options,
                                   family = "poisson")
model_poisson_tf %>% fit(epochs = 1000, validation_split = 0)

model_poisson_torch <- deepregression(y = y,
                                      list_of_formulas = list( rate = ~1 + x),
                                   data = data_poisson,
                                   family = "poisson",
                                   engine = "torch",
                                   orthog_options = orthog_options)

model_poisson_torch %>% fit(epochs = 1000, validation_split = 0)


res_poisson <- cbind("torch" = unlist(model_poisson_torch %>% coef()),
                      "tf" = unlist(model_poisson_tf %>% coef()),
                      "glm" = rev(coef(glm(formula = y~x, family = poisson))),
                     "true" = c(2, -0.5))
res_poisson
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

# beta = rate = 1/scale
# alpha = shape (concentration)
gamlss_gamma <- gamlss(formula = y ~ 1 + x, sigma.formula = y~1, 
                       family = "GA", data = data_gamma)
-coef(gamlss_gamma)
glm_gamma <- glm(formula = y~x, 
                 data = data_gamma, family = Gamma(link = "log"))
-coef(glm_gamma)

model_gamma_tf <- deepregression(y = y,
                                    list_of_formulas = list(concentration = ~ 1,
                                                            rate = ~ 1 + x),
                                    data = data_gamma,
                                    family = "gamma",
                                    engine = "tf",
                                    orthog_options = orthog_options)

model_gamma_tf %>% fit(epochs = 1000, validation_split = 0)
model_gamma_tf %>% coef(which_par = 1)
model_gamma_tf %>% coef(which_par = 2)

model_gamma_torch <- deepregression(y = y,
                                    list_of_formulas = list(
                                      concentration = ~ 1 ,
                                      rate = ~ 1+ x),
                                    data = data_gamma,
                                    family = "gamma",
                                    engine = "torch",
                                    orthog_options = orthog_options,
                                    subnetwork_builder = subnetwork_init_torch,
                                    model_builder = torch_dr)

model_gamma_torch %>% fit(epochs = 1000, validation_split = 0)
model_gamma_torch %>% coef(which_par = 1)
model_gamma_torch %>% coef(which_par = 2)

res_gamma <- list(
  "concentration" = cbind("torch" = unlist(model_gamma_torch %>% coef()),
                          "tf" = unlist(model_gamma_tf %>% coef()),
                          "gamlss" = rev(coef(gamlss_gamma, 'sigma'))),
  "rate" = cbind("torch" = unlist(model_gamma_torch %>% coef(2)),
                              "tf" = unlist(model_gamma_tf %>% coef(2)),
                              "gamlss" = rev(-coef(gamlss_gamma, 'mu'))))

# bernoulli
set.seed(42)
n <- 1000
x <- runif(n, -1, 1)
# plogis == inverse logit
y <- rbinom(n = length(x), size = 1, prob = plogis(1 + 2*x))
data_bern <- data.frame(x)
glm(y ~ 1 + x, family = "binomial")

model_bernoulli_tf <- deepregression(y = y,
                                       list_of_formulas = list( 
                                         logit = ~ 1 + x),
                                       data = data_bern,
                                       family = "bernoulli",
                                       engine = "tf",
                                       orthog_options = orthog_options)
model_bernoulli_tf %>% fit(epochs = 500, validation_split = 0.2)
model_bernoulli_torch <- deepregression(y = y,
                                      list_of_formulas = list( 
                                        logit = ~ 1 + x),
                                      data = data_bern,
                                      family = "bernoulli",
                                      engine = "torch",
                                      orthog_options = orthog_options,
                                      subnetwork_builder = subnetwork_init_torch,
                                      model_builder = torch_dr)
model_bernoulli_torch %>% fit(epochs = 100, validation_split = 0.2)
model_bernoulli_torch %>% coef()

res_binomial <- cbind("torch" = unlist(model_bernoulli_torch %>% coef()),
                      "tf" = unlist(model_bernoulli_tf %>% coef()),
                      "glm" = rev(coef(glm(y ~ 1 + x, family = "binomial"))))
res_binomial
      