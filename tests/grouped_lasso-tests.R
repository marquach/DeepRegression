#grouped lasso test
# New setup
# try to compare with gglasso package
set.seed(42)
tf$random$set_seed(42)
n <- 1000
x <- sample(1:10, size = 1000, replace = T)
x <- as.factor(x)
x <- data.frame(x)

x_help <- dummyVars(" ~ .", data = x)
x_onehot <- data.frame(predict(x_help, newdata = x))
beta <- c(0,10,0,rep(0,7))

y <- as.matrix(x_onehot) %*% beta  + rnorm(n)

mod_tf <- deepregression(y = y, data = x,
                         list_of_formulas = 
                           list(
                             location = ~ -1 + grlasso(x, la  = 1),
                             scale = ~1),
                         orthog_options = orthog_control(orthogonalize = F),
                         engine = "tf")
mod_tf$model$input
mod_tf %>% fit(epochs = 1000, validation_split = 0)
mod_tf %>% coef()

library(gglasso)

# group lasso

gr <-  gglasso(model.matrix(~ ., x), as.vector(y), lambda = 2,
               loss="ls", intercept = F)
gr$beta