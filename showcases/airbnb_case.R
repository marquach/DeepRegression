# Case Study Airbnb
library(torch)
library(luz)
library(torchvision)
devtools::load_all("deepregression-main/")
source('scripts/deepregression_functions.R')
orthog_options = orthog_control(orthogonalize = F)


airbnb <- readRDS("/Users/marquach/Desktop/R_Projects/semi-structured_distributional_regression/application/airbnb/munich_clean.RDS")

airbnb$days_since_last_review <- as.numeric(
   difftime(airbnb$date, airbnb$last_review)
   )

list_of_formulas = list(
   loc = ~ 1 + x + s(z1, bs="tp"),
   scale = ~ 0 + s(z2, bs="ps") + dnn(u)
   )

y = log(airbnb$price)

list_of_formulas = list(
   loc = ~ 1 + te(latitude, longitude, df = 5),
   scale = ~ 1
   )

mod_simple <- deepregression(y = y, data = airbnb,
                             list_of_formulas = list_of_formulas,
                             list_of_deep_models = NULL)

deep_model_tf <- function(x){
    x %>%
     layer_dense(units = 5, activation = "relu", use_bias = FALSE) %>%
     layer_dropout(rate = 0.2) %>%
     layer_dense(units = 3, activation = "relu") %>%
     layer_dropout(rate = 0.2) %>%
     layer_dense(units = 1, activation = "linear")
}

deep_model_torch <- function() nn_sequential(
  nn_linear(in_features = 2, out_features = 5, bias = F),
  nn_relu(),
  nn_dropout(p = 0.2),
  nn_linear(in_features = 5, out_features = 3, bias = F),
  nn_relu(),
  nn_dropout(p = 0.2),
  nn_linear(in_features = 3, out_features = 1))


options(identify_intercept = TRUE)

mod_tf <- deepregression(y = y, data = airbnb,
                      list_of_formulas = 
                        list(
                          location = ~ 1 + beds + s(accommodates, bs = "ps") +
                            s(days_since_last_review, bs = "tp") +
                            deep(review_scores_rating, reviews_per_month),
                          scale = ~1),
                      orthog_options = orthog_options,
                      list_of_deep_models = list(deep = deep_model_tf),
                      engine = "tf"
   )

mod_torch <- deepregression(y = y, data = airbnb,
                         list_of_formulas = 
                           list(
                             location = ~ 1 + beds +  s(accommodates, bs = "ps") +
                               s(days_since_last_review, bs = "tp") +
                               deep(review_scores_rating, reviews_per_month),
                             scale = ~1), orthog_options = orthog_options,
                         list_of_deep_models = list(deep = deep_model_torch),
                         engine = "torch",
                         subnetwork_builder = subnetwork_init_torch,
                         model_builder = torch_dr
                         )
mod_tf %>% fit( epochs = 100, validation_split = 0.2)
mod_torch %>% fit(epochs = 100, validation_split = 0.2)

fitted_vals_tf <- mod_tf %>% fitted()
fitted_vals_torch <- mod_torch %>% fitted()


cor(data.frame(fitted_vals_tf, fitted_vals_torch, y))

coef(mod_tf, type="linear")
coef(mod_torch, type="linear")

plot(mod_tf)
plot(mod_torch)

dist_tf <- mod_tf %>% get_distribution()
dist_torch <- mod_torch %>% get_distribution()
str(dist_tf$tensor_distribution, 1)
str(dist_torch, 1)


first_obs_airbnb <- as.data.frame(airbnb)[1,,drop=F]
dist1_tf <- mod_tf %>% get_distribution(first_obs_airbnb)
dist1_torch <- mod_torch %>% get_distribution(first_obs_airbnb)

meanval_tf <- mod_tf %>% mean(first_obs_airbnb) %>% c
meanval_torch <- mod_torch %>% mean(first_obs_airbnb) %>% c

q05_tf <- mod_tf %>% quant(data = first_obs_airbnb, probs = 0.05) %>% c
q95_tf <- mod_tf %>% quant(data = first_obs_airbnb, probs = 0.95) %>% c

q05_torch <- mod_torch %>% quant(data = first_obs_airbnb, probs = 0.05) %>% c
q95_torch <- mod_torch %>% quant(data = first_obs_airbnb, probs = 0.95) %>% c

xseq <- seq(q05_tf-1, q95_tf+1, l=1000)
plot(xseq, sapply(xseq, function(x) c(
  as.matrix(dist1_tf$tensor_distribution$prob(x)))),
     type="l", ylab = "density(price)", xlab = "price")
abline(v = c(q05_tf, meanval_tf, q95_tf), col="red", lty=2)

xseq <- seq(q05_torch-1, q95_torch+1, l=1000)
plot(xseq, sapply(xseq, function(x) c(
  as.matrix(exp(dist1_torch$log_prob(x))))),
  type="l", ylab = "density(price)", xlab = "price")
abline(v = c(q05_tf, meanval_tf, q95_tf), col="red", lty=2)



# working with images
airbnb$image <- paste0("/Users/marquach/Desktop/R_Projects/semi-structured_distributional_regression/application/airbnb/data/pictures/32/",
                       airbnb$id, ".jpg")

cnn_block_tf <- function(
     filters,
     kernel_size,
     pool_size,
     rate,
     input_shape = NULL
     ){
   function(x){
     x %>%
       layer_conv_2d(filters, kernel_size,
                       padding="same", input_shape = input_shape) %>%
       layer_activation(activation = "relu") %>%
       layer_batch_normalization() %>%
       layer_max_pooling_2d(pool_size = pool_size) %>%
       layer_dropout(rate = rate)
     }
}

cnn_block_torch <- function(
    filters,
    kernel_size,
    pool_size,
    rate,
    shape = NULL
){
  function() nn_sequential(
    #layer_conv_2d(filters, kernel_size, padding="same", input_shape = input_shape)
    nn_conv2d(in_channels = 3, out_channels = filters,
              kernel_size = kernel_size, padding = "same"),
    nn_relu(),
    nn_batch_norm2d(num_features = filters),
    nn_max_pool2d(kernel_size = kernel_size),
    nn_dropout(p = rate)
    )
}
cnn_torch <- cnn_block_torch(
  filters = 16,
  kernel_size = c(3,3),
  pool_size = c(3,3),
  rate = 0.25,
  shape(200, 200, 3)
)
cnn_torch()

cnn_tf <- cnn_block_tf(
   filters = 16,
   kernel_size = c(3,3),
   pool_size = c(3,3),
   rate = 0.25,
   shape(200, 200, 3)
   )

deep_model_cnn_tf <- function(x){
   x %>%
     cnn_tf() %>%
     layer_flatten() %>%
     layer_dense(32) %>%
     layer_activation(activation = "relu") %>%
     layer_batch_normalization() %>%
     layer_dropout(rate = 0.5) %>%
     layer_dense(1)
}

deep_model_cnn_torch <- function(){
  nn_sequential(
    cnn_torch(),
    nn_flatten(),
    nn_linear(in_features = 69696, out_features = 32),
    nn_relu(),
    nn_batch_norm1d(num_features = 32),
    nn_dropout(p = 0.5),  
    nn_linear(32, 1) )
}
deep_model_cnn_torch()

mod_cnn_tf <- deepregression(
   y = y,
   list_of_formulas = list(
     ~-1 + deep_model_cnn(image),
     ~1),
   data = airbnb,
   list_of_deep_models = list(deep_model_cnn =
                                  list(deep_model_cnn_tf, c(200,200,3))))

mod_cnn_torch <- deepregression(
  y = y,
  list_of_formulas = list(
    ~ -1  + deep_model_cnn(image),
    ~1 ),
  data = airbnb,
  engine = "torch",
  subnetwork_builder = subnetwork_init_torch,
  model_builder = torch_dr, orthog_options = orthog_options,
  list_of_deep_models = list(deep_model_cnn =
                               list(deep_model_cnn_torch, c(200,200,3))))

mod_cnn_tf %>% fit(
   epochs = 60, batch_size = 56,
   early_stopping = F)

mod_cnn_torch %>% fit(
  epochs = 60, batch_size = 56,
  early_stopping = F)

fitted_tf <- mod_cnn_tf %>% fitted()
fitted_torch <- mod_cnn_torch %>% fitted()
plot(fitted_torch, fitted_tf)
cor(fitted_torch, fitted_tf)

library(luz)
library(torchvision)

model_t <- deep_model_cnn_torch()
optimizer <- optim_adam(model_t$parameters)

get_image_dataset <- dataset(
  "deepregression_image_dataset",
  
  initialize = function(images, target) {
    self$image <- images
    self$target <- target
  },
  
  .getitem = function(index) {
    
    indexes <- self$image[index] %>% base_loader() %>%
      torchvision::transform_to_tensor()
    
    target <- self$target[index]
    list(indexes, target)
  },
  
  .length = function() {
    length(self$target)
  }
  
)

ds <- get_image_dataset(airbnb$image, y)
ds$.getitem(1)

test_dl <- ds %>% dataloader(batch_size = 20)

luz_model <- nn_module(
  initialize = function(){
    self$cnn <- cnn_torch()
    self$fc <- nn_linear(in_features = 69696, out_features = 32)
    self$batchnorm1d <- nn_batch_norm1d(num_features = 32)
    self$fc2 <- nn_linear(32, 1)
  },
  forward = function(x){
    x %>% self$cnn() %>% torch_flatten(start_dim = 2) %>% self$fc() %>%
      nnf_relu() %>% self$batchnorm1d() %>% nnf_dropout(p = 0.5) %>%
      self$fc2() %>% torch_flatten()
  }
  )

pre_fit <- luz_model %>% luz::setup(loss = nnf_mse_loss,
                         optimizer = optim_adam
                         )
fitte <- pre_fit %>% fit(test_dl, epochs = 1)
