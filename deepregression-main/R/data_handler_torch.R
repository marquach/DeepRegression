prepare_data_torch <- function(pfc, input_x, target = NULL){
  
  distr_datasets_length <- sapply(
    1:length(pfc),
    function(x) length(pfc[[x]])
  )
  
  sequence_length <- seq_len(sum(distr_datasets_length))
  
  distr_datasets_index <- lapply(distr_datasets_length, function(x){
    index <- sequence_length[seq_len(x)]
    sequence_length <<- sequence_length[-seq_len(x)]
    index
  })
  
  df_list <- lapply(distr_datasets_index, function(x){ input_x[x] })
  
  if(is.null(target)) return(df_list)
  
  get_luz_dataset(df_list = df_list, target = torch_tensor(target)$to(torch_float()))
  }

prepare_input_list_model <- function(input_x, input_y,
                                     object, epochs = 10, batch_size = 32,
                                     validation_split = 0, validation_data = NULL,
                                     callbacks = NULL, verbose, view_metrics, 
                                     early_stopping){
  
  if(object$engine == "torch"){
    input_dataloader <- prepare_data_torch(
      pfc  = object$init_params$parsed_formulas_content,
      input_x = input_x,
      target = input_y)
    
    #no validation
    if(is.null(validation_data) & identical(validation_split, 0)){
      train_dl <- dataloader(input_dataloader, batch_size = batch_size)
      valid_dl <- NULL
    }
    
    if(!is.null(validation_data)){
      train_dl <- dataloader(input_dataloader, batch_size = batch_size)
      
      validation_dataloader <- prepare_data_torch(
        pfc  = object$init_params$parsed_formulas_content,
        input_x = validation_data[[1]],
        target = validation_data[[2]])
      
      valid_dl <- dataloader(validation_dataloader, batch_size = batch_size)
    }
    
    if(!identical(validation_split, 0) & !is.null(validation_split)){
      train_ids <- sample(1:length(input_y), 
                          size = ceiling((1-validation_split) * length(input_y)))
      valid_ids <- setdiff(1:length(input_y), train_ids)
      valid_ids <- sample(valid_ids,
                          size = length(valid_ids))
      
      train_ds <- dataset_subset(input_dataloader, indices = train_ids)
      valid_ds <- dataset_subset(input_dataloader, indices = valid_ids)
      
      if(any(unlist(lapply(input_x, check_data_for_image)))) 
        cat(sprintf("Found %s validated image filenames \n", length(train_ids)))
      train_dl <- dataloader(train_ds, batch_size = batch_size)
      if(any(unlist(lapply(input_x, check_data_for_image))))
        cat(sprintf("Found %s validated image filenames \n", length(valid_ids)))
      valid_dl <- dataloader(valid_ds, batch_size = batch_size)
    }
    
    if(!is.null(valid_dl)) valid_data <- valid_dl
    if(is.null(valid_dl)) valid_data <- NULL
    
    input_list_model <- list(
      object = object$model,
      epochs = epochs,
      data = train_dl,
      valid_data = valid_data,
      callbacks = callbacks,
      verbose = verbose)
    return(c(input_list_model))
  }
  
  if(object$engine == 'tf'){
    input_list_model <-
      list(object = object$model,
           epochs = epochs,
           batch_size = batch_size,
           validation_split = validation_split,
           validation_data = validation_data,
           callbacks = callbacks,
           verbose = verbose,
           view_metrics = ifelse(view_metrics, getOption("keras.view_metrics", default = "auto"), FALSE)
      )
    
    input_list_model <- c(input_list_model,
                          list(x = input_x, y = input_y)
    )
    
    input_list_model}
}