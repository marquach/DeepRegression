
library(gamlss)
devtools::load_all("deepregression-main/")

test_that("Poisson Distributins works", {

    set.seed(42)
    n <- 1000
    x <- sample(x = log(85:125), size = n, replace = T)
    mu <- -0.5 + 2*x
    y <- rpois(n = n, lambda = exp(mu))
    data_poisson <- data.frame(x)
    
    
    model_poisson_torch <- deepregression(y = y,
                                          list_of_formulas = list( rate = ~1 + x),
                                       data = data_poisson,
                                       family = "poisson",
                                       engine = "torch")
    
    expect_true(model_poisson_torch$init_params$family == "poisson")
    expect_equal(length(model_poisson_torch$model()[[1]]), 1)
    
    model_poisson_torch %>% fit(epochs = 1000, validation_split = 0)
    
    res_poisson <- cbind("torch" = unlist(model_poisson_torch %>% coef()),
                          "glm" = rev(coef(glm(formula = y~x, family = poisson))),
                         "true" = c(2, -0.5))
 
    expect_equal(res_poisson[,"torch"],
                 res_poisson[,"glm"], tolerance = 0.1)
})

test_that("Gamma (no natural link) Distributins works", {
  set.seed(42)
  n <- 1000
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
  glm_gamma <- glm(formula = y~x, 
                 data = data_gamma, family = Gamma(link = "log"))
  


  model_gamma_torch <- deepregression(y = y,
                                    list_of_formulas = list(
                                      concentration = ~ 1 ,
                                      rate = ~ 1+ x),
                                    data = data_gamma,
                                    family = "gamma",
                                    engine = "torch")
  model_gamma_torch %>% fit(epochs = 1000, validation_split = 0)
  
  res_gamma <- list(
    "concentration" = cbind("torch" = unlist(model_gamma_torch %>% coef()),
                            "gamlss" = rev(coef(gamlss_gamma, 'sigma'))),
    "rate" = cbind("torch" = unlist(model_gamma_torch %>% coef(2)),
                    "gamlss" = rev(-coef(gamlss_gamma, 'mu'))))
  expect_equal(
    res_gamma[[1]][,"torch"],
    res_gamma[[1]][,"gamlss"], tolerance = 0.1)
  expect_equal(
    res_gamma[[2]][,"torch"],
    res_gamma[[2]][,"gamlss"], tolerance = 0.1)

})


test_that("Bernoulli Distributions works", {

  set.seed(42)
  n <- 1000
  x <- runif(n, -1, 1)
  # plogis == inverse logit
  y <- rbinom(n = length(x), size = 1, prob = plogis(1 + 2*x))
  data_bern <- data.frame(x)
  
  model_bernoulli_torch <- deepregression(y = y,
                                        list_of_formulas = list( 
                                          logit = ~ 1 + x),
                                        data = data_bern,
                                        family = "bernoulli",
                                        engine = "torch")
  model_bernoulli_torch %>% fit(epochs = 250, validation_split = 0)
  model_bernoulli_torch %>% coef()
  
  res_binomial <- cbind("torch" = unlist(model_bernoulli_torch %>% coef()),
                        "glm" = rev(coef(glm(y ~ 1 + x, family = "binomial"))))

  expect_equal(
    res_binomial[,"torch"],
    res_binomial[,"glm"], tolerance = 0.1)
  
  })
      