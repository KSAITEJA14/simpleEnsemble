% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Bagging.R
\name{bag}
\alias{bag}
\title{Comprehensive Bagging Function for Various Regression and Classification Models}
\usage{
bag(
  X.train,
  y.train,
  X.test,
  y.test = NULL,
  model.type,
  R,
  alpha = NULL,
  lambda = NULL,
  intercept = TRUE,
  kernel = NULL,
  imp.threshold = NULL,
  combine.method = "stacking"
)
}
\arguments{
\item{model.type}{A character string specifying the type of model to be fitted. Supported model types include "linear", "logistic", "ridge", "lasso", "elastic", and "svm". This parameter determines the type of predictive model that will be applied to the input data. Each model type is handled according to its own specific implementation within the function.}

\item{R}{The number of bootstrapping samples.
This parameter is crucial when implementing methods that require
repeated sampling to estimate model stability or performance.}

\item{alpha}{This parameter controls the mixing between lasso (L1) and ridge (L2) penalties.
`alpha = 1` is lasso, while `alpha = 0` is ridge regression.
Typically ranges between 0 and 1.}

\item{lambda}{The regularization parameter in ridge, lasso, and elastic net models.
Controls the amount of shrinkage: the larger the lambda, the more shrinkage,
thus driving coefficients to zero in lasso and towards zero in ridge.}

\item{intercept}{logical value indicating whether the model should include an intercept term.
Default is `TRUE`, which includes an intercept in the model. Set to `FALSE` to fit the model
without an intercept, which may be appropriate in certain contexts, such as when data are centered.}

\item{kernel}{Specifies the type of kernel to be used in kernel-based learning algorithms, such as SVMs.
Common choices include "linear", "polynomial", "radial", and "sigmoid". Each kernel type can
influence the decision boundary and performance differently based on the nature of the data.}

\item{imp.threshold}{The importance threshold for feature selection.
Features with importance measures (like variable importance in random forests) below this threshold
may be excluded from the model. This helps in reducing model complexity and overfitting.}

\item{combine.method}{The method to be used for combining model predictions. Supported methods are "average" for simple averaging of predictions and "stacking" for combine predictions based on their performance using Meta learner which is an SVM.}

\item{X_train}{A matrix of predictor variables for training the models.}

\item{y_train}{A vector of the response variable used for training models.}

\item{X_test}{A matrix of predictor variables for testing the models.}

\item{y_test}{A vector of the response variable used for evaluating the models}
}
\value{
A vector of predictions resulting from the specified method of combining model outputs.
}
\description{
This function implements a generic bagging algorithm which can be applied to different model types including linear, logistic, ridge, lasso, and elastic-net models. It handles feature importance and uses different methods for combining predictions.
}
\examples{
#' {
  # Using the mtcars dataset for a logistic regression model example
  X <- as.matrix(mtcars[, 1:7])
  y <- as.integer(mtcars$am)  # am as a binary response variable
  result1 <- bag(X, y, model.type = "logistic", R = 10)

  # Using the iris dataset for a linear regression model example
  X <- as.matrix(iris[, 1:4])
  y <- iris$Sepal.Length
  result2 <- bag(X, y, model.type = "linear", R = 10)
}

}
\seealso{
\code{\link[stats]{lm}}, \code{\link[stats]{glm}}
}
