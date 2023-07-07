# Script to test generic functions

# first start with tabular data only
# showcase from the deepregression (intro) presentation
set.seed(42)
n <- 1000
data = data.frame(matrix(rnorm(4*n), c(n,4)))
colnames(data) <- c("x1","x2","x3","xa")
y <- rnorm(n) + data$xa^2 + data$x1

formula <- ~ 1 + deep_model(x1,x2,x3) + s(xa) + x1

nn_tf <- function(x) x %>%
  layer_dense(units = 32, activation = "relu", use_bias = FALSE) %>%
  layer_dense(units = 1, activation = "linear")

nn_torch <- nn_module(
  initialize = function(){
    self$fc1 <- nn_linear(in_features = 3, out_features = 32, bias = F)
    self$fc2 <-  nn_linear(in_features = 32, out_features = 1)
  },
  forward = function(x){
    x %>% self$fc1() %>% nnf_relu() %>% self$fc2()
  }
)

semi_structured_tf <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y,orthog_options = orthog_options,
  list_of_deep_models = list(deep_model = nn_tf), engine = "tf"
)

semi_structured_torch <- deepregression(
  list_of_formulas = list(loc = formula, scale = ~ 1),
  data = data, y = y, orthog_options = orthog_options,
  list_of_deep_models = list(deep_model = nn_torch),
  subnetwork_builder = subnetwork_init_torch, model_builder = torch_dr,
  engine = "torch"
)

# Summary
semi_structured_tf
semi_structured_torch

semi_structured_tf %>% fit(epochs = 100, early_stopping = F, batch_size = 256)
semi_structured_torch %>% fit(epochs = 100,
                              early_stopping = F, batch_size = 256)

plot(semi_structured_tf %>% fitted(),
     semi_structured_torch %>% fitted(),
     xlab = "tensorflow", ylab = "torch")

cor(semi_structured_tf %>% fitted(),
    semi_structured_torch %>% fitted())

# coef

# works with defaults
cbind(
  "tf" = unlist(semi_structured_tf %>% coef()),
  "torch" = unlist(semi_structured_torch %>% coef()))


# works with type
cbind(unlist(semi_structured_tf %>% coef(type = "smooth")),
      unlist(semi_structured_torch %>% coef(type = "smooth")))

cbind(unlist(semi_structured_tf %>% coef(type = "linear")),
      unlist(semi_structured_torch %>% coef(type = "linear")))

# works with which_param
coef(semi_structured_torch, which_param = 1)
coef(semi_structured_tf, which_param = 1)
coef(semi_structured_torch, which_param = 2)
coef(semi_structured_tf, which_param = 2)

predict(semi_structured_torch, newdata = data[1,])
fitted(semi_structured_torch)[1,]
predict(semi_structured_torch, newdata = data[100,])
fitted(semi_structured_torch)[100,]


semi_structured_tf %>% cv(epochs = 2)
semi_structured_torch %>% cv(epochs = 2)

