
# not really fast (torch 0.10 and luz 0.4 are now faster)
fast_fit_function <- function(object, epochs, batch_size, data_x, data_y,
                              validation_split){
  
  data_x <- lapply(data_x, FUN = function(x) lapply(x, torch_tensor))
  data_y <- torch_tensor(data_y)$to(torch_long())
  
  num_data_points <- data_y$size(1)
  
  train_ids <- sample(1:num_data_points,
                      size = (1-validation_split) * num_data_points)
  valid_ids <- sample(
    setdiff(1:num_data_points,train_ids),
    size = validation_split * num_data_points)
  

  data_x_train <- lapply(data_x, function(x) lapply(x ,
                                                    function(x) x[train_ids,]))
  data_x_valid <- lapply(data_x, function(x) lapply(x ,
                                                    function(x) x[valid_ids,]))
  data_y_train <- data_y[train_ids]
  data_y_valid <- data_y[valid_ids]
  
  num_batches_train <- floor(data_y_train$size(1)/batch_size)
  num_batches_valid <- floor(data_y_valid$size(1)/batch_size)
  
  optimizer_t_manual <- optim_adam(object$model()$parameters)
  
  for(epoch in 1:epochs){
    object$model()$train()
    l_man <- c()
    # rearrange the data each epoch
    permute <- torch_randperm(data_y_train$size(1)) + 1L
    data_x_train <- lapply(data_x_train, function(x) lapply(x ,
                                                            function(x) x[permute,]))
    data_y_train <- data_y_train[permute]
    
    # manually loop through the batches
    for(batch_idx in 1:num_batches_train){
      
      # here index is a vector of the indices in the batch
      index <- (batch_size*(batch_idx-1) + 1):(batch_idx*batch_size)
      
      x <- lapply(data_x_train, function(x) lapply(x ,
                                                  function(x) x[index,]))
      y <- data_y_train[index]
      
      optimizer_t_manual$zero_grad()
      
      output <- object$model()(x)
      loss_man <- object$model()$loss(output, y)
      loss_man$backward()
      
      optimizer_t_manual$step()
      l_man <- c(l_man, loss_man$item())
    }
    cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l_man)))
    
    object$model()$eval()
    valid_loss <- c()
    
    with_no_grad({
    # manually loop through the batches
    for(batch_idx in 1:num_batches_valid){
      
      # here index is a vector of the indices in the batch
      index <- (batch_size*(batch_idx-1) + 1):(batch_idx*batch_size)
      
      x <- lapply(data_x_valid, function(x) lapply(x ,
                                                    function(x) x[index,]))
      y <- data_y_valid[index]
      
      output <- object$model()(x)
      loss_val <- object$model()$loss(output, y)
      
      valid_loss <- c(valid_loss, loss_val$item())
    }
    cat(sprintf("Valid loss at epoch %d: %3f\n", epoch, mean(valid_loss)))
    
  })
  }
}







