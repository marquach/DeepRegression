library(torch)
torch_manual_seed(42)
x <- torch_arange(0, 1, step = 0.001)$view(c(1001,1))
#y <- 2*x + 1 + torch_randn(100, 1) # mean von y
intercept <- 1
beta1 <- 2

n <- distr_normal(loc = beta1*x, scale = torch_exp(beta1*x))
y <- n$sample(1)
plot(as.numeric(x), as.numeric(y))

GaussianLinear <- nn_module(
  initialize = function() {
    # this linear predictor will estimate the mean of the normal distribution
    self$linear <- nn_linear(1, 1, bias = F)
    # this parameter will hold the estimate of the variability
    self$scale <- nn_linear(1, 1, bias = F)
  },
  forward = function(x) {
    # we estimate the mean
    loc <- self$linear(x)
    scale <-  self$scale(x)
    # return a normal distribution
    #distr_normal(loc, self$scale)
    distr_normal(loc, torch_exp(scale))
    
  }
)

model <- GaussianLinear()
model$parameters

opt <- optim_adam(model$parameters)

for (i in 1:500) {
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
(2.011^2)
