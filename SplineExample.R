#Example for background
library(ggpubr)
set.seed(123)

x  = runif(500)
mu = sin(2 * (4 * x - 2)) + 2 * exp(-(16 ^ 2) * ((x - .5) ^ 2))
y  = rnorm(500, mu, .3)
knots = seq(0, 1, by = .1)

bfs = splines::bs(x, knots = knots[-1])
bsMod = lm(y ~ bfs)
fits = fitted(bsMod)
bfsScaled = sweep(cbind(Int = 1, bfs), 2, coef(bsMod), `*`)


bSplineXmatsc = data.frame(x, fits, bfsScaled) %>%
  tidyr::pivot_longer(-c(x, fits), names_to = 'bs', values_to = 'bsfunc')

bSplineXmat = data.frame (Int = 1, x = x, bfs) %>%
  tidyr::pivot_longer(-c(x), names_to = 'bs', values_to = 'bsfunc')

basis_plot <- d %>%
  ggplot(aes(x, y)) +
  geom_point(color = 'black', alpha = .1) +
  geom_line(
    aes(y = bsfunc, color = bs),
    linewidth = 1,
    data = bSplineXmat,
    show.legend = FALSE
  ) + theme_classic()




basis_plot_weighted <- d %>%
  ggplot(aes(x, y)) +
  geom_point(color = 'black', alpha = .1) +
  geom_line(
    aes(y = bsfunc, color = bs),
    size = 1,
    data = bSplineXmatsc,
    show.legend = FALSE
  ) + theme_classic()

ggarrange(basis_plot, basis_plot_weighted)

output <- tibble(x,
                 y,
                 "true function" = mu,
                   mgcv = fitted(gam(y ~ s(x, bs = 'cr'), knots = list(x = knots[-1])))) %>%
                      tidyr::pivot_longer(-c(x, y), names_to = 'mod', values_to = 'pred')  %>%
                      # mutate(pred = fitted(lmModCubicSpline)) %>%
                      ggplot(aes(x, y)) +
                      geom_point(color = 'black', alpha = .1) +
                      geom_line(
                        aes(y = pred, color = mod),
                        linewidth = 2,
                        alpha = .75, 
                        show.legend = T
                      )+theme_classic()
output
