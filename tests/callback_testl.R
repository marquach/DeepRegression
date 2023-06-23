
mse_train <- 0
mse_valid <- 0

calc_mse_callback <- luz_callback(
  name = "mse_callback",
  initialize = function() {
    self$mse_train <- 0
    self$mse_valid <- 0
  },
  on_train_end = function() {
    self$mse_train <- nnf_mse_loss(input = ctx$pred$mean, ctx$target)
    mse_train <<- self$mse_train
    print(mse_train)
  }
  on_valid_end = function() {
    self$self$mse_valid <- nnf_mse_loss(input = ctx$pred$mean, ctx$target)
    self$mse_valid <<- self$self$mse_valid
    print(self$mse_valid)
  }
)

mod_torch_fitted <- mod_torch %>% fit(epochs = 2, validation_split = 0.2,
                                      callbacks = list(calc_mse_callback()))
