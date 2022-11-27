# Deep Regression example

library(deepregression)
set.seed(42)
n <- 1000
data = data.frame(matrix(rnorm(4*n), c(n,4)))
colnames(data) <- c("x1","x2","x3","xa")
formula <- ~ 1 + deep_model(x1,x2,x3) + s(xa) + x1

deep_model <- function(x) x %>%
  layer_dense(units = 32, activation = "relu", use_bias = FALSE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 8, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")

y <- rnorm(n) + data$xa^2 + data$x1

debugonce(deepregression)  
# Hiermit kann man sich schön durch die Funktion klicken. Muss aber jedesmal davor
# ausgeführt werden, sonst funktioniert es nicht. 
# debug() muss nur einmal ausgeführt werden, aber man muss das debug() dann auch
# wieder schließen sonst passiert es jedesmal bei der Funktion.
# https://fort-w2021.github.io/week4.html
# Da gehts um das debugging

mod <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y,
  list_of_deep_models = list(deep_model = deep_model)
)

if(!is.null(mod)){
  
  # train for more than 10 epochs to get a better model
  mod %>% fit(epochs = 10, early_stopping = TRUE)
  mod %>% fitted() %>% head()
  cvres <- mod %>% cv()
  mod %>% get_partial_effect(name = "s(xa)")
  mod %>% coef()
  mod %>% plot()
  
}

mod <- deepregression(
  list_of_formulas = list(loc = ~ 1 + s(xa) + x1, scale = ~ 1,
                          dummy = ~ -1 + deep_model(x1,x2,x3) %OZ% 1),
  data = data, y = y,
  list_of_deep_models = list(deep_model = deep_model),
  mapping = list(1,2,1:2)
)
