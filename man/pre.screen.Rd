% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Preprocessing.R
\name{pre.screen}
\alias{pre.screen}
\title{Top K features}
\usage{
pre.screen(X, y, k, method)
}
\arguments{
\item{X}{A matrix of predictor variables.}

\item{y}{A vector of the response variable.}

\item{k}{The number of features to select.}

\item{method}{The method used for feature selection, supported methods are "correlation" for continuous response, "chi-square" for categorical response, and "pca" for any type of response.}
}
\value{
A list containing two elements: `selected.indices` which are the indices of the selected features, and `selected.data` which is the subset of X corresponding to the selected features.
}
\description{
This function performs feature selection based on specified statistical methods. It supports three methods: correlation, chi-square, and PCA. The function is designed to handle different data types and select the most relevant features based on the method and the response variable characteristics.
}
\examples{
{
  Example using a hypothetical dataset
  X <- matrix(rnorm(100 * 10), ncol = 10)  # 100 observations and 10 features
  y <- rbinom(100, 1, 0.5)  # Binary outcome
  result <- pre.screen(X, y, k = 5, method = "correlation")

  # Example using 'chi-square' method, requires y as factor
  y_factor <- factor(sample(c("Group1", "Group2"), 100, replace = TRUE))
  result_chi <- pre.screen(X, y_factor, k = 3, method = "chi-square")

  # PCA method example
  result_pca <- pre.screen(X, y, k = 4, method = "pca")
}

}
\seealso{
\code{\link[stats]{cor}}, \code{\link[stats]{chisq.test}}, \code{\link[stats]{prcomp}}
}
