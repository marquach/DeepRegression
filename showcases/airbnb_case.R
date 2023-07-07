# Case Study Airbnb
# https://arxiv.org/pdf/2104.02705.pdf#page=30&zoom=100,72,725
library(torch)
library(luz)
library(torchvision)
devtools::load_all("deepregression-main/")
source('scripts/deepregression_functions.R')

orthog_options = orthog_control(orthogonalize = F)

# load  and prepare data
airbnb <- readRDS("/Users/marquach/Desktop/R_Projects/semi-structured_distributional_regression/application/airbnb/munich_clean.RDS")
airbnb$days_since_last_review <- as.numeric(
   difftime(airbnb$date, airbnb$last_review)
   )
y = log(airbnb$price)

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
  nn_linear(in_features = 5, out_features = 3, bias = T),
  nn_relu(),
  nn_dropout(p = 0.2),
  nn_linear(in_features = 3, out_features = 1))

mod_tf <- deepregression(y = y, data = airbnb,
                      list_of_formulas = 
                        list(
                          location = ~ 1 + beds +
                            s(accommodates, bs = "ps") +
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
                             location = ~ 1 + beds +
                               s(accommodates, bs = "ps") +
                               s(days_since_last_review, bs = "tp") +
                               deep(review_scores_rating, reviews_per_month),
                             scale = ~1), orthog_options = orthog_options,
                         list_of_deep_models = list(deep = deep_model_torch),
                         engine = "torch"
                         )

mod_tf %>% fit(epochs = 25, validation_split = 0.2, verbose = T)
mod_torch %>% fit(epochs = 10, validation_split = 0.2)

fitted_vals_tf <- mod_tf %>% fitted()
fitted_vals_torch <- mod_torch %>% fitted()

cor(data.frame(fitted_vals_tf, fitted_vals_torch, y))
plot(fitted_vals_tf, fitted_vals_torch)
abline(a = 0,b = 1)

coef(mod_tf)
coef(mod_torch)

cbind(unlist(coef(mod_tf, type="linear")),
      unlist(coef(mod_torch, type="linear")))

coef(mod_tf, type="smooth")
coef(mod_torch, type="smooth")

cbind(coef(mod_tf, type="smooth")[[1]],
      coef(mod_torch, type="smooth")[[1]])

cbind(coef(mod_tf, type="smooth")[[2]],
      coef(mod_torch, type="smooth")[[2]])

coef(mod_tf, which_param = 1)
coef(mod_torch, which_param = 1)

coef(mod_tf, which_param = 2)
coef(mod_torch, which_param = 2)

plot(mod_tf, which = 2)
mod_torch_data <- plot(mod_torch,  which = 2, only_data = T)
points(mod_torch_data[[1]]$value, mod_torch_data[[1]]$partial_effect, 
       col = "green")

# Cross validation
res_cv_tf <- mod_tf %>% cv(plot = F, cv_folds = 3, epochs = 10)
res_cv_torch <- mod_torch %>% cv(plot = F, cv_folds = 3, epochs = 10)

dist_tf <- mod_tf %>% get_distribution()
dist_torch <- mod_torch %>% get_distribution()
str(dist_tf$tensor_distribution, 1)
attributes(dist_torch)


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

# shared dnn

# load  and prepare data
airbnb_texts <- readRDS("/Users/marquach/Desktop/R_Projects/semi-structured_distributional_regression/application/munich_clean_text.RDS")
airbnb_texts$days_since_last_review <- as.numeric(
  difftime(airbnb$date, airbnb$last_review)
)
y = log(airbnb_texts$price)

airbnb_texts_torch <- airbnb_texts
airbnb_texts_torch$texts <- airbnb_texts_torch$texts+1L 

embd_mod_tf <- function(x) x %>%
  layer_embedding(input_dim = 1000,
                  output_dim = 100) %>%
  layer_lambda(f = function(x) k_mean(x, axis = 2)) %>%
  layer_dense(20, activation = "tanh") %>%
  layer_dropout(0.3) %>%
  layer_dense(2)

shared_dnn_tf <- deepregression(
  y = y,
  list_of_formulas = list(
    location = ~ 1,
    scale = ~ 1,
    both = ~ 0 + embd_mod(texts)
  ),
  mapping = list(1, 2, 1:2),
  list_of_deep_models = list(embd_mod = embd_mod_tf),
  data = airbnb_texts
)

embd_mod_torch <- nn_module(
  classname = "embd_torch_test",
  initialize = function(){
    self$embedding <- nn_embedding(num_embeddings =  1000,
                                   embedding_dim = 100)
    self$fc1 <-   nn_linear(in_features = 100, out_features = 20)
    self$fc2 <-   nn_linear(in_features = 20, out_features = 2)
  },
  forward = function(x){
    x %>% self$embedding() %>%  torch_mean(dim = 2) %>%
      self$fc1() %>% torch_tanh() %>% nnf_dropout(0.3) %>% self$fc2()
    
  })

shared_dnn_torch <- deepregression(
  y = y,
  list_of_formulas = list(
    location = ~ 1,
    scale = ~ 1,
    both = ~ 0 + embd_mod(texts)),
  engine = "torch",
  mapping = list(1, 2, 1:2),
  list_of_deep_models = list(embd_mod = embd_mod_torch),
  orthog_options = orthog_options,
  data = airbnb_texts_torch
)

shared_dnn_tf %>% fit(epochs = 10, validation_split = 0.1)
shared_dnn_torch %>% fit(epochs = 10, validation_split = 0.1)

plot(shared_dnn_tf %>% fitted(),
     shared_dnn_torch %>% fitted()
)
do.call(what = plot, 
        list(shared_dnn_tf %>% fitted(),
             shared_dnn_torch %>% fitted(),
             xlab = "", ylab = ""))
do.call(what = cor, 
        list(shared_dnn_tf %>% fitted(),
             shared_dnn_torch %>% fitted()))
abline(a = 0, b = 1)

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

mod_cnn_tf <- deepregression(
   y = y,
   list_of_formulas = list(
     ~1 + room_type + bedrooms + beds +
       deep_model_cnn(image),
      ~1 + room_type),
   data = airbnb,
   list_of_deep_models = list(deep_model_cnn =
                                  list(deep_model_cnn_tf, c(200,200,3))))

mod_cnn_torch <- deepregression(
  y = y,
  list_of_formulas = list(
    ~1 + room_type + bedrooms + beds +
       deep_model_cnn(image),
     ~1 + room_type),
  data = airbnb,
  engine = "torch",
  orthog_options = orthog_options,
  list_of_deep_models = list(deep_model_cnn =
                               list(deep_model_cnn_torch, c(200,200,3))))


mod_cnn_tf %>% fit(
   epochs = 2, batch_size = 56,
   early_stopping = F)

mod_cnn_torch %>% fit(
  epochs = 2, batch_size = 56,
  early_stopping = F)


fitted_tf <- mod_cnn_tf %>% fitted()
fitted_torch <- mod_cnn_torch %>% fitted()
plot(fitted_torch, fitted_tf)
cor(fitted_torch, fitted_tf)

mod_cnn_tf %>% predict(airbnb[1,])
fitted_tf[1]
mod_cnn_torch %>% predict(airbnb[1,])
fitted_torch[1]


