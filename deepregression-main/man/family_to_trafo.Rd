% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/families.R
\name{family_to_trafo}
\alias{family_to_trafo}
\title{Character-to-transformation mapping function}
\usage{
family_to_trafo(family, add_const = 1e-08)
}
\arguments{
\item{family}{character defining the distribution}

\item{add_const}{see \code{\link{make_tfd_dist}}}
}
\value{
a list of transformation for each distribution parameter
}
\description{
Character-to-transformation mapping function
}
