set.seed(24)
n <- 500
x <- runif(n) %>% as.matrix()
z <- runif(n) %>% as.matrix()

y <- x - z
data <- data.frame(x = x, z = z, y = y)

mod_tf <- deepregression(
  y = y,
  data = data,
  list_of_formulas = list(loc = ~ 1 + x + z, scale = ~ 1),
  list_of_deep_models = NULL,
  family = "normal",
  from_preds_to_output = function(x, ...) x[[1]],
  loss = "mse"
)
mod_tf
fit(mod_tf, epochs = 1000)

model_mse <- function(submodules_list){
  nn_module(
    classname = "model_mse",
    initialize = function() {
      self$subnetwork <- lapply(submodules_list, function(x)
        nn_module_dict(x))[[1]]
    },
    
    forward = function(dataset_list) {
      subnetworks <- lapply(1:length(self$subnetwork$children), function(x){
        self$subnetwork[[x]](dataset_list[[1]][[x]])
      })
      
      Reduce(f = torch_add, subnetworks)
    }
  )}

options(orthogonalize = F)
mod_torch <- deepregression(
  y = y,
  data = data,
  list_of_formulas = list(loc = ~ 1 + x + z, scale = ~ 1),
  list_of_deep_models = NULL,
  family = "normal", engine = "torch",
  from_preds_to_output = function(x, ...) model_mse(x),
  loss = nnf_mse_loss
  )

mod_torch$model()[[1]]

mod_torch$model()$parameters
mod_tf %>% coef()
fit(mod_torch, epochs = 10)
coef(mod_torch)




################################################################################
################################################################################
#################### additive models #################### 
#### GAM data mcycles
data(mcycle, package = 'MASS')
plot(mcycle$times, mcycle$accel)
# Erst mit mgcv


gam_mgcv <- gam(mcycle$accel ~ 1 + s(times), data = mcycle)
gam_data <- model.matrix(gam_mgcv)
gam_tf <- deepregression(
  list_of_formulas = list(loc = ~ 1 + s(times), scale = ~ 1),
  data = mcycle, y = mcycle$accel,
  engine = "tf",
  from_preds_to_output = function(x, ...) x[[1]],
  loss = "mse"
)
gam_torch <- deepregression(
  list_of_formulas = list(loc = ~ 1 + s(times), scale = ~ 1),
  data = mcycle, y = mcycle$accel,
  engine = "torch",
  from_preds_to_output = function(x, ...) model_mse(x),
  loss = nnf_mse_loss)

# adapt learning rate to converge faster
gam_torch$model <- gam_torch$model  %>% set_opt_hparams(lr = 0.5)
gam_tf$model$optimizer$lr <- tf$Variable(0.5, name = "learning_rate")

gam_torch %>% fit(epochs = 20, early_stopping = F, validation_split = 0)
gam_tf %>% fit(epochs = 20, early_stopping = F, validation_split = 0)
# Interesting (does not converge when validation_loss is used as early_stopping)

mean((mcycle$accel - fitted(gam_mgcv) )^2)
# loss is not MSE so loss of model not (directly) comparable
# Loss in deepregression is neg.loglikehood of given distribution
mean((mcycle$accel - gam_tf %>% fitted(apply_fun =  function(x) x))^2)
mean((mcycle$accel - gam_torch %>% fitted(apply_fun =  function(x) x))^2)


plot(mcycle$times, mcycle$accel)
lines(mcycle$times, fitted(gam_mgcv), col = "red")
lines(mcycle$times, gam_tf %>% fitted(apply_fun =  function(x) x), col = "green")
lines(mcycle$times, gam_torch  %>% fitted(apply_fun =  function(x) x),
      col = "blue")

cbind( 
  "gam" = c(coef(gam_mgcv)[2:10],coef(gam_mgcv)[1]) ,
  "tf" = unlist(gam_tf %>% coef()),
  "torch" = unlist(gam_torch$model()$parameters %>% coef()))

gam_torch %>% coef()
