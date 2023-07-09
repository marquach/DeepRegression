library(rbenchmark)
library(ggplot2)
devtools::load_all("deepregression-main/")

plain_loop_fit_function <- function(model, epochs, batch_size, data_x, data_y,
                                    validation_split, verbose = F, shuffle = T){
  
  data_x <- torch_tensor(as.matrix(data_x))
  data_y <- torch_tensor(data_y)
  num_data_points <- data_y$size(1)
  data_y <- data_y$view(c(data_y$size(1),1))
  
  train_ids <- sample(1:num_data_points,
                      size = (1-validation_split) * num_data_points)
  valid_ids <- sample(
    setdiff(1:num_data_points,train_ids),
    size = validation_split * num_data_points)
  
  
  data_x_train <- data_x[train_ids,]
  data_x_valid <- data_x[valid_ids,]
  
  data_y_train <- data_y[train_ids]
  data_y_valid <- data_y[valid_ids]
  
  num_batches_train <- floor(data_y_train$size(1)/batch_size)
  num_batches_valid <- floor(data_y_valid$size(1)/batch_size)
  
  optimizer_t_manual <- optim_adam(model$parameters)
  
  for(epoch in 1:epochs){
    
    model$train()
    l_man <- c()
    
    # rearrange the data each epoch
    if(shuffle){
    permute <- torch_randperm(data_y_train$size(1)) + 1L
    data_x_train <- data_x_train[permute,]
    data_y_train <- data_y_train[permute]
    }
    
    # manually loop through the batches
    for(batch_idx in 1:num_batches_train){
      
      # here index is a vector of the indices in the batch
      index <- (batch_size*(batch_idx-1) + 1):(batch_idx*batch_size)
      
      x <- data_x_train[index,]
      y <- data_y_train[index]
      
      optimizer_t_manual$zero_grad()
      
      output <- model(x)
      loss_man <- nnf_mse_loss(input = output, target = y)
      loss_man$backward()
      optimizer_t_manual$step()
      
      
      l_man <- c(l_man, loss_man$item())
    }
    if(verbose) cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l_man)))
    
    model$eval()
    valid_loss <- c()
    
    with_no_grad({
      # manually loop through the batches
      for(batch_idx in 1:num_batches_valid){
        
        # here index is a vector of the indices in the batch
        index <- (batch_size*(batch_idx-1) + 1):(batch_idx*batch_size)
        
        x <- data_x_valid[index,]
        y <- data_y_valid[index]
        
        output <- model(x)
        loss_val <- nnf_mse_loss(input = output, target = y)
        
        valid_loss <- c(valid_loss, loss_val$item())
      }
      if(verbose) cat(sprintf("Valid loss at epoch %d: %3f\n", epoch, mean(valid_loss)))
      
    })
    
    
  }
}

set.seed(42)
n = c(500, 1000, 2000)
epochs <- c(100, 500, 1000)

settings <- expand.grid(n = n, epochs = epochs)
setting_names <- lapply(1:nrow(settings),
                        function(x)
                          paste("n =",paste(settings[x,], collapse = ", epochs = ")
                                ))
formula <- ~ 1 + x 

setting_res <- lapply(X = 1:nrow(settings), function(x){
  
  #setting combination
  n <- settings[x,1]
  epochs <- settings[x,2]
  
  # data generation
  x <- runif(n)
  y <- rnorm(n = n, mean = 2*x, sd = 1)  + 1
  data <- data.frame(x = x)
  
  semi_structured_torch <- deepregression(
    list_of_formulas = list(loc = formula, scale = ~ 1),
    data = data, y = y, orthog_options = orthog_options,
    from_preds_to_output = function(x, ...) x[[1]],
    loss = nnf_mse_loss,
    engine = "torch")
  
  semi_structured_tf <- deepregression(
    list_of_formulas = list(loc = formula, scale = ~ 1),
    data = data, y = y, orthog_options = orthog_options,
    from_preds_to_output = function(x, ...) x[[1]],
    loss = "mse",
    engine = "tf")
  
  # model for loop
  data_plain <- data
  data_plain <- cbind(1, data_plain)
  
  plain_torch_loop <- nn_module(
    initialize = function(){
      self$linear <- nn_linear(in_features = 1, out_features = 1, T)
    },
    
    forward = function(input){
      self$linear(input[,2, drop = F])
    }
  )
  
  model <- plain_torch_loop()
  
  # setup for luz
  # same as in deepregression (initialized outside)
  intercept_loc <- layer_dense_torch(input_shape = 1, units = 1)
  linear <- layer_dense_torch(input_shape = 1, units = 1)
  
  plain_torch_luz <- nn_module(
    initialize = function(){
      self$intercept_loc <- intercept_loc
      self$linear <- linear
    },
    forward = function(input){
      self$intercept_loc(input[,1, drop = F] ) + self$linear(input[,2, drop = F])
    }
  )
  plain_luz <- luz::setup(plain_torch_luz, 
                          loss = nnf_mse_loss,
                          optimizer = optim_adam)
  
  # adapt learning rate to same as loop
  plain_luz <- plain_luz %>% set_opt_hparams(lr = 0.1)
  semi_structured_torch$model <- semi_structured_torch$model  %>%
    set_opt_hparams(lr = 0.1)
  semi_structured_tf$model$optimizer$lr <- 
    tf$Variable(0.1, name = "learning_rate")
  
  
  plain_deepreg_semistruc_benchmark <- benchmark(
    "torch_plain" = plain_loop_fit_function(model, epochs = epochs, 
                                            batch_size = 32,
                                            data_x = data_plain, data_y = y,
                                            validation_split = 0.1, shuffle = F),
    "torch_luz" = plain_luz_fitted <- fit(plain_luz,
                                          data = list(as.matrix(data_plain), 
                                                      as.matrix(y)),
                                          epochs = epochs, verbose = F,
                                          valid_data = 0.1,
                                          dataloader_options = 
                                            list(batch_size = 32)),
    "deepregression_luz" = semi_structured_torch %>%
      fit(epochs = epochs,
          early_stopping = F,
          validation_split = 0.1,
          verbose = F, batch_size = 32), 
    "deepregression_tf" = semi_structured_tf %>% fit(epochs = epochs, 
                                                     early_stopping = F,
                                                     validation_split = 0.1, verbose = F,
                                                     batch_size = 32),
    columns = c("test", "elapsed", "relative"),
    order = "elapsed",
    replications = 2
  )
  
  results <- cbind(
    "plain_loop" = rev(lapply(model$parameters, as.array)),
    "plain_luz" = lapply(plain_luz_fitted$model$parameters, as.array),
    "deepreg_torch" = rev(coef(semi_structured_torch)),
    "deepreg_tf" = rev(coef(semi_structured_tf,1)),
    "gamlss" =coef(gamlss::gamlss(y~x, data = data)))
  
  list("benchmark"= plain_deepreg_semistruc_benchmark,
       "results"= results)
})

names(setting_res) <- setting_names
setting_res

save_name <- sprintf("scripts/speed_comparison/speed_comparison_mse_%s.RData", Sys.Date())
save(setting_res, file = save_name)

setting_res_frame <- Reduce(x = lapply(setting_res, function(x) x[[1]]),
                            rbind)
setting_res_frame$n <- unlist(lapply(1:nrow(settings), function(x)
  rep(settings[x,1], 4))) # 4 different approaches
setting_res_frame$epochs <- unlist(lapply(1:nrow(settings), function(x)
  rep(settings[x,2], 4)))
setting_res_frame$test <- factor(setting_res_frame$test, levels = 
                                   c("deepregression_tf",
                                     "torch_plain",
                                     "torch_luz",
                                     "deepregression_luz"), ordered = T)
levels(setting_res_frame$test) <- c("Deepreg TF", "Torch", "Torch Luz",
                                    "Deepreg torch")
colnames(setting_res_frame)[1] <- c("Approaches")

ggplot(data = setting_res_frame, aes(x = Approaches, y = elapsed, fill = Approaches))+
  geom_bar(stat = "identity", position = "dodge2")+
  facet_grid(n ~ epochs,
             labeller = labeller(
               epochs = c("100" = "epochs = 100",
                          "500" = "epochs = 500",
                          "1000" = "epochs = 1000"),
               n = c("500" = "n = 500",
                     "1000" = "n = 1000",
                     "2000" = "n = 2000")
             ))+
  theme_classic()+ ylab("Elapsed Time (s)") + xlab("Approaches")+
  scale_x_discrete(guide = guide_axis(n.dodge = 2))+
  geom_text(data = setting_res_frame, aes(x = Approaches, y = elapsed,
                                          label = round(relative,2)),
            position = position_stack(vjust = 0.5))

