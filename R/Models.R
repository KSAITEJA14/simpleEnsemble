#' Fit Predictive Models with Custom Parameters
#'
#'
#' This function allows fitting of various types of models including linear, logistic, ridge, lasso, elastic net, and SVM. It supports custom configurations for model parameters like alpha, lambda, and kernel.
#'
#'
#' @param X A matrix of predictor variables.
#' @param y A vector of the response variable.
#' @param model.type A character string specifying the type of model to be fitted. Supported model types include "linear", "logistic", "ridge", "lasso", "elastic", and "svm".
#' @param alpha The mixing parameter, it must be between 0 and 1. Default is 0.5.
#' @param lambda The regularization parameter for ridge, lasso, and elastic net models. Default is 0.0001.
#' @param intercept Logical, indicates whether an intercept should be included in the model. Default is TRUE.
#' @param kernel The type of kernel to be used in SVM models. Supported kernels are "linear", "polynomial", "radial", and "sigmoid". Default is "radial".
#'
#' @return The fitted model object, which varies depending on the model type specified. For linear and logistic regression, returns a `glm` object. For ridge, lasso, and elastic net, returns a `glmnet` object. For SVM, returns an `e1071::svm` object.
#'
#' @export
#'
#'
#' @examples
#' {
#'   # Example using mtcars dataset
#'   X <- as.matrix(mtcars[, 1:7])
#'   y <- mtcars$mpg  # Using mpg as the response variable
#'   model <- fit.model(X, y, model.type = "ridge")
#'
#'   # Example with logistic regression using iris dataset
#'   X <- as.matrix(iris[, 1:4])
#'   y <- ifelse(iris$Species == "setosa", 1, 0)  # Binary outcome
#'   logistic_model <- fit.model(X, y, model.type = "logistic")
#'}
#'@seealso \code{\link[glmnet]{glmnet}}, \code{\link[e1071]{svm}}
#'
fit.model <- function(X, y, model.type, alpha = NULL, lambda = NULL, intercept = TRUE, kernel = NULL) {
  if (is.null(alpha)) {
    if (model.type %in% c("ridge","linear","logistic","svm")) { alpha <- 0}
    else if (model.type == "lasso") {alpha <- 1}
    else if(model.type == "elastic") {
      alpha <- 0.5 #Default for elastic net
    }
  }

  if (is.null(lambda)) {
    if (model.type %in% c("lasso", "ridge", "elastic")) {
      # Perform cross-validation to find the optimal lambda
      cv_fit <- cv.glmnet(X, y, alpha = alpha, family = (ifelse(length(unique(y)) == 2, "binomial", "gaussian")))
      lambda <- cv_fit$lambda.min
    }
    else if (model.type %in% c("linear","logistic","svm")) {lambda <- 0}
  }

  if (is.null(kernel)) kernel <- "radial"  # Default kernel for SVM

  # Error Handling
  if (!is.matrix(X)) {
    stop("Error: X must be a matrix.")
  }

  if (!is.vector(y) || length(dim(y)) > 1) {
    stop("Error: y must be a vector.")
  }

  if (!is.null(alpha) && (alpha < 0 || alpha > 1)) {
    stop("Error: alpha should be between 0 and 1.")
  }

  # Determine if y is binary
  unique_y <- unique(y)
  is_binary_response <- length(unique_y) == 2 && all(unique_y %in% c(0, 1))
  valid.models <- c("linear", "logistic", "ridge", "lasso", "elastic", "svm")

  if (!(model.type %in% valid.models)) {
    stop("Error: Invalid model type. Supported types are: linear, logistic, ridge, lasso, elastic, svm")
  }

  if (! (kernel %in% c("linear", "polynomial", "radial", "sigmoid"))) {
    stop("Error: kernel should be any of ('linear', 'polynomial', 'radial', 'sigmoid')")
  }

  if (!intercept && lambda == 0 && model.type == "lasso") {
    stop("Error: lambda must be greater than 0 when model has no intercept in lasso regression")
  }

  # Model Fitting
  model_fit <- NULL  # Initialize variable for model object

  if (is_binary_response) {
    if (model.type %in% c("ridge", "lasso", "elastic", "logistic")) {
      # Use glmnet with binomial family for binary outcomes with regularization
      model_fit <- glmnet::glmnet(X, y, family = "binomial", alpha = alpha, lambda = lambda, intercept = intercept)
    } else if (model.type == "svm") {
      # Use SVM for classification
      model_fit <- e1071::svm(x = X, y = y, type = "C-classification", kernel = kernel)
    } else if (model.type == "linear") {
      stop("Linear regression model does not support binary response")
    }
  }
  else {
    if (model.type %in% c("ridge", "lasso", "elastic", "linear")) {
      # Use glmnet for continuous outcomes with regularization
      model_fit <- glmnet::glmnet(X, y, family = "gaussian", alpha = alpha, lambda = lambda, intercept = intercept)
      }
    else if (model.type == "svm") {
      # Use SVM for regression
      model_fit <- e1071::svm(x = X, y = y, type = "eps-regression", kernel = kernel)
    } else {
      stop("Unsupported model type for continuous response")
    }
  }

  return(model_fit)
}
