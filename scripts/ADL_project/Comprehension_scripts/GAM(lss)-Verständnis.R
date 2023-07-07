
#Skript zum Verstehen von GAM LSS und GAM

library(mgcv)
library(luz)
library(torch)

#get data
set.seed(42)
n <- 1000

data = data.frame(matrix(rnorm(4*n), c(n,4)))
colnames(data) <- c("x1","x2","x3","xa")
y <- rnorm(n) + data$xa^2 + data$x1

# Test
debugonce(gam)
gam_model <- gam(y ~ 1 + x1 + s(xa), data = data, family = gaussian)
gam_model$coefficients
round(gam_model$smooth[[1]]$S[[1]])
gam_model$sp
gam_model$full.sp


library(deepregression)
formula <- ~ 1 +s(xa)
debugonce(deepregression)  
mod <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y, orthog_options = orthog_options,
  model_builder = model_builder_test, engine = 'torch'
)






dat<-gamlss.dist::rNO(100)
dat
debugonce(gamlss)
library(gamlss)
gamlss(dat~1,family="NO") # fits a constant for mu and sigma 

