# Deep Regression example
library(deepregression)
set.seed(42)
n <- 1000
data = data.frame(matrix(rnorm(4*n), c(n,4)))
colnames(data) <- c("x1","x2","x3","xa")
formula <- ~ 1 + deep_model(x1,x2,x3) + s(xa) + x1

# deep_model: list of DNNs for deepregression 
# 20% dropout rate 
deep_model <- function(x) x %>%
  layer_dense(units = 32, activation = "relu", use_bias = FALSE) %>%
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 8, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")

y <- rnorm(n) + data$xa^2 + data$x1

browser()
debugonce(deepregression)  
# Hiermit kann man sich schön durch die Funktion klicken. Muss aber jedesmal davor
# ausgeführt werden, sonst funktioniert es nicht. 
# debug() muss nur einmal ausgeführt werden, aber man muss das debug() dann auch
# wieder schließen sonst passiert es jedesmal bei der Funktion.
# https://fort-w2021.github.io/week4.html
# Da gehts um das debugging
# debugonce() kann auch innerhalb wieder über Konsole genutzt werden.
# z.b. debugonce(precalc_gam)

mod_1 <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y,
  list_of_deep_models = list(deep_model = deep_model)
)

if(!is.null(mod_1)){
  
  # train for more than 10 epochs to get a better model
  mod %>% fit(epochs = 10, early_stopping = TRUE)
  mod %>% fitted() %>% head()
  cvres <- mod %>% cv()
  mod %>% get_partial_effect(name = "s(xa)")
  mod %>% coef()
  mod %>% plot()
  
}

mod_1_coef <- coef(mod_1)

# nur mit dummys  extra -> ergibt die gleichen koeffizienten 
mod_2 <- deepregression(
  list_of_formulas = list(loc = ~ 1 + s(xa) + x1, scale = ~ 1,
                          dummy = ~ -1 + deep_model(x1,x2,x3) %OZ% 1),
  data = data, y = y,
  list_of_deep_models = list(deep_model = deep_model),
  mapping = list(1,2,1:2)
)


mod_2_koef <- coef(mod_2)
mod_2_koef[[1]]-mod_1_coef[[1]] #gleich s(x)


# vergleiche mit Gam aus mgcv
library(mgcv)
model_gam <- gam(y~1 + x2+ x3 + s(xa) + x1, data=data)
model_gam_koef <- coef(model_gam)
model_gam_koef


### verstehe was passiert 
debug(deepregression(list_of_formulas = list(loc = formula, scale = ~ 1), data = data, y = y, list_of_deep_models = list(deep_model = deep_model)))
debug(deepregression)
browser(deepregression(list_of_formulas = list(loc = formula, scale = ~ 1),
                       data = data, y = y,
                       list_of_deep_models = list(deep_model = deep_model)))


debug(deepregression(list_of_formulas = list(loc = formula, scale = ~ 1),
                     data = data, y = y,
                     list_of_deep_models = list(deep_model = deep_model)))














