devtools::load_all("deepregression-main/")
source("scripts/deepregression_functions.R")
set.seed(42)
tf$random$set_seed(42)
n <- 1000
x <- sample(1:3, size = 1000, replace = T)
x <- as.factor(x)
x <- data.frame(x)

x_help <- dummyVars(" ~ .", data = x)
x_onehot <- data.frame(predict(x_help, newdata = x))
beta <- c(2,5,2)

y <- as.matrix(x_onehot) %*% beta  + rnorm(n)

mod_tf <- deepregression(y = y, data = x,
                            list_of_formulas = 
                              list(
                                location = ~ 1 + x,
                                scale = ~1),
                            orthog_options = orthog_control(orthogonalize = F),
                            engine = "tf")

mod_torch <- deepregression(y = y, data = x,
                            list_of_formulas = 
                              list(
                                location = ~ 1 + x,
                                scale = ~1),
                            orthog_options = orthog_control(orthogonalize = F),
                            engine = "torch",
                            subnetwork_builder = subnetwork_init_torch,
                            model_builder = torch_dr
)

mod_tf %>% fit(epochs = 300, validation_split = 0.2)
mod_torch %>% fit(epochs = 300, validation_split = 0.2)

# true betas are 2,5,2

mod_tf %>% coef()
# beta1 = 2
(mod_tf %>% coef())[[2]] 
# beta2 = 5 
(mod_tf %>% coef())[[2]] + (mod_tf %>% coef())[[1]][1,1]
# beta3 = 2
(mod_tf %>% coef())[[2]] + (mod_tf %>% coef())[[1]][2,1]

mod_torch %>% coef()
# beta1 = 2
(mod_torch %>% coef())[[2]] 
# beta2 = 5 
(mod_torch %>% coef())[[2]] + (mod_torch %>% coef())[[1]][1,1]
# beta3 = 2
(mod_torch %>% coef())[[2]] + (mod_torch %>% coef())[[1]][2,1]

# so it does work

# but if we leave the intercept out of the formula we run into problems
mod_tf <- deepregression(y = y, data = x,
                         list_of_formulas = 
                           list(
                             location = ~ -1 + x,
                             scale = ~1),
                         orthog_options = orthog_control(orthogonalize = F),
                         engine = "tf")
mod_torch <- deepregression(y = y, data = x,
                            list_of_formulas = 
                              list(
                                location = ~ -1 + x,
                                scale = ~1),
                            orthog_options = orthog_control(orthogonalize = F),
                            engine = "torch",
                            subnetwork_builder = subnetwork_init_torch,
                            model_builder = torch_dr
)

#only two parameters for mu (both models)
# should be three because we droped intercept  (=no dummy var. trap)
mod_tf$model$weights[[1]]
mod_torch$model()$parameters[[1]]




