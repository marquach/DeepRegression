# Case Study Airbnb
# https://arxiv.org/pdf/2104.02705.pdf#page=30&zoom=100,72,725
library(torch)
library(luz)
library(torchvision)
devtools::load_all("deepregression-main/")
source('scripts/deepregression_functions.R')

orthog_options = orthog_control(orthogonalize = F)

# load  and prepare data
airbnb_texts <- readRDS("/Users/marquach/Desktop/R_Projects/semi-structured_distributional_regression/application/munich_clean_text.RDS")

airbnb_texts$days_since_last_review <- as.numeric(
  difftime(airbnb$date, airbnb$last_review)
)
y = log(airbnb_texts$price)

embd_mod <- function(x) x %>%
  layer_embedding(input_dim = 1000,
                   output_dim = 100) %>%
  layer_lambda(f = function(x) k_mean(x, axis = 2)) %>%
  layer_dense(20, activation = "tanh") %>%
  layer_dropout(0.3) %>%
  layer_dense(1)

nlp_mod_tf <- deepregression(
   y = y,
   list_of_formulas = list(
     location = ~ -1 + embd_mod(texts),
     scale = ~ 1
     ),engine = "tf",
   list_of_deep_models = list(embd_mod = embd_mod),
   data = airbnb_texts
   )
nlp_mod_tf %>% fit(epochs = 10, validation_split = 0.2)
tf_nlp <- nlp_mod_tf %>% fitted()

embd_mod_torch <- nn_module(
  classname = "embd_torch_test",
  initialize = function(){
    self$embedding <- nn_embedding(num_embeddings =  1000,
                                   embedding_dim = 100)
    self$fc1 <-   nn_linear(in_features = 100, out_features = 20)
    self$fc2 <-   nn_linear(in_features = 20, out_features = 1)
    },
  forward = function(x){
    x %>% self$embedding() %>%  torch_mean(dim = 2) %>%
      self$fc1() %>% torch_tanh() %>% nnf_dropout(0.3) %>% self$fc2()

  })

airbnb_texts_torch <- airbnb_texts
airbnb_texts_torch$texts <- airbnb_texts_torch$texts+1L  

nlp_mod_torch <- deepregression(
  y = y,
  list_of_formulas = list(
    location = ~ -1 + embd_mod(texts),
    scale = ~ 1
  ),engine = "torch",
  list_of_deep_models = list(embd_mod = embd_mod_torch),
  orthog_options = orthog_options,
  data = airbnb_texts_torch,
  subnetwork_builder = subnetwork_init_torch,
  model_builder = torch_dr
)

nlp_mod_torch %>% fit(epochs = 10, validation_split = 0.2)
torch_nlp <- nlp_mod_torch %>% fitted()
plot(tf_nlp, torch_nlp)


# Shared dnn test for nlp
embd_mod <- function(x) x %>%
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
  list_of_deep_models = list(embd_mod = embd_mod),
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
  data = airbnb_texts_torch,
  subnetwork_builder = subnetwork_init_torch,
  model_builder = torch_dr
)

shared_dnn_tf %>% fit(epochs = 100, validation_split = 0)
shared_dnn_tf %>% fitted()

shared_dnn_torch %>% fit(epochs = 100, validation_split = 0)
shared_dnn_torch %>% fitted()


plot(shared_dnn_tf %>% fitted(),
     shared_dnn_torch %>% fitted()
)


