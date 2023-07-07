# data for tests

# First a toy example with known true coefficients
x <- seq(from = 0, to = 1, 0.001)
beta1 <- 2
set.seed(42)
y <- rnorm(n = 1001, mean = 0*x, sd = exp(beta1*x))
plot(x,y)
toy_data <- data.frame(x = x, y = y)

set.seed(42)
n <- 1000
data = data.frame(matrix(rnorm(4*n), c(n,4)))
colnames(data) <- c("x1","x2","x3","xa")
y <- rnorm(n) + data$xa^2 + data$x1
