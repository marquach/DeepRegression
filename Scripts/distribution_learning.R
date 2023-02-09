library(torch)
torch_manual_seed(42)
x <- torch_arange(0, 1, step = 0.001)$view(c(1001,1))
beta1 <- 2

n <- distr_normal(loc = beta1*x, scale = torch_exp(beta1*x))
y <- n$sample(1)
plot(as.numeric(x), as.numeric(y))


# hier brauchen wir lapply for initialize
# Output ist bei allen 1
# input ist abhängig davon wie viele Terme in Formula pro Verteilungsparameter

learn_gaussian_distr <- nn_module(
  initialize = function() {
    self$linear <- nn_linear(in_features = 1, out_features = 1, bias = F)
    self$scale <- nn_linear(1, 1, bias = F)
  },
  forward = function(x) {
    loc <- self$linear(x)
    scale <-  self$scale(x)
    # hier lapply, da Anzahl der Parameter unterschiedlich
    # distr_normal, je nach Verteilung
    distr_normal(loc, torch_exp(scale))
  }
)

model <- learn_gaussian_distr()

opt <- optim_adam(model$parameters)

for (i in 1:200) {
  opt$zero_grad()
  d <- model(x)
  loss <- torch_mean(-d$log_prob(y))
  loss$backward()
  opt$step()
  if (i %% 10 == 0)
    cat("iter: ", i, " loss: ", loss$item(), "\n")
}

model$parameters

gamlss(formula = as.numeric(y)~ -1+as.numeric(x),
       sigma.formula = ~ -1 + as.numeric(x), family = NO)
