#' Ensemble Learning for Model Combination
#'
#' This function constructs an ensemble of various predictive models specified by the user and combines their predictions using either averaging or SVM-based stacking. It supports multiple types of models, including linear regression, logistic regression, ridge regression, lasso regression, elastic net, and support vector machines.
#'
#'
#' @param X_train A matrix of predictor variables for training the models.
#' @param y_train A vector of the response variable used for training models.
#' @param X_test A matrix of predictor variables for testing the models.
#' @param y_test A vector of the response variable used for evaluating the models
#' @param model.specs A list of lists, where each inner list specifies the model type and its parameters. For a given model, user must always specify the parameters as NULL if they are not used in the model. Example: list(list(model="linear", alpha = NULL, lambda = NULL,intercept = TRUE, kernel = NULL), list(model="logistic", alpha = NULL, lambda = NULL,intercept = TRUE, kernel = NULL)).
#' @param combine.method The method to be used for combining model predictions. Supported methods are "average" for simple averaging of predictions and "stacking" for combine predictions based on their performance using Meta learner which is an SVM.
#'
#' @return A vector of combined predictions based on the specified combination method.
#' @export
#'
#' @examples
#' {
#'   # Example using mtcars dataset
#'   X <- as.matrix(mtcars[, 1:7])
#'   y <- mtcars$mpg  # Using mpg as the response variable
#'   specs <- list(list(model="linear"), list(model="ridge", lambda=0.1))
#'   predictions <- ensemble(X, y, specs, combine.method = "average")
#'
#'   # Using SVM stacking
#'   specs <- list(list(model="linear"), list(model="lasso", lambda=0.05))
#'   predictions <- ensemble(X, y, specs, combine.method = "svm_stacking")
#'}
#'@seealso \code{\link[stats]{lm}}, \code{\link[stats]{glm}}, \code{\link[stats]{svm}}
ensemble<- function(X.train, y.train, X.test, y.test = NULL, model.specs, combine.method = "stacking") {
  # Error Handling
  if (!is.matrix(X.train) || !is.matrix(X.test)) {
    stop("Error: X_train and X_test must be matrices.")
  }
  if(!(ncol(X.train) == ncol(X.test))){
    stop("Error: X_train and X_test must have same number of features.")
  }
  if (!is.vector(y.train)) {
    stop("Error: y_train must be a vector.")
  }
  if (!is.null(y.test) && !is.vector(y.test)) {
    stop("Error: y_test must be a vector.")
  }

  X.train = na.omit(X.train)
  X.test = na.omit(X.test)

  valid.models <- c("linear", "logistic", "ridge", "lasso", "elastic", "svm")
  if (!all(sapply(model.specs, function(x) x$model %in% valid.models))) {
    stop("Error: One of more unsupported models.")
  }
  unique_y <- unique(y.train)
  is_binary_response <- length(unique_y) == 2 && all(unique_y %in% c(0, 1))

  # Initialize matrices for predictions
  train_prediction.matrix <- matrix(NA, nrow = nrow(X.train), ncol = length(model.specs))
  test_prediction.matrix <- matrix(NA, nrow = nrow(X.test), ncol = length(model.specs))

  # Model Fitting
  model.list <- list()
  for (i in seq_along(model.specs)) {
    spec <- model.specs[[i]]
    op.model <- do.call(fit.model, c(list(X = X.train, y = y.train), spec))
    if (spec[1] %in% c("ridge", "lasso", "elastic", "logistic", "linear")){
      train_prediction.matrix[, i] <- predict(op.model, as.matrix(X.train))
      test_prediction.matrix[, i] <- predict(op.model, as.matrix(X.test))
    }
    else {
      train_prediction.matrix[, i] <- predict(op.model, as.data.frame(X.train))
      test_prediction.matrix[, i] <- predict(op.model, as.data.frame(X.test))
    }
  }

  # Helper function to combine predictions based on specified method
  combine_predictions <- function(predictions, y, method, is_binary) {
    if (is_binary) {
      probabilities <- 1 / (1 + exp(-predictions))
      binary_predictions <- ifelse(probabilities > 0.5, 1, 0)
      if (method == "majority") {
        apply(binary_predictions, 1, function(x) { which.max(tabulate(match(x, unique(x)))) })
      } else { # Stacking using SVM
        meta.model <- fit.model(predictions, y, model.type = "svm", kernel = "linear")
        predict(meta.model, newdata = as.data.frame(predictions))
      }
    } else {
      if (method == "average") {
        rowMeans(predictions, na.rm = TRUE)
      } else { # Stacking using SVM
        meta.model <- fit.model(predictions, y, model.type = "svm", kernel = "linear")
        predict(meta.model, newdata = as.data.frame(predictions))
      }
    }
  }

  # Combine predictions using the specified method
  final.train.predictions <- combine_predictions(train_prediction.matrix, y.train, combine.method, is_binary_response)
  final.test.predictions <- combine_predictions(test_prediction.matrix, y.test, combine.method, is_binary_response)

  if (is_binary_response) {
    combined.train.accuracy <- mean(final.train.predictions == y.train)
    combined.test.accuracy <- if (!is.null(y.test)) mean(final.test.predictions == y.test) else NA
  } else { # Calculate combined RMSE for continuous response
    combined.train.accuracy <- sqrt(mean((final.train.predictions - y.train)^2))
    combined.test.accuracy <- if (!is.null(y.test)) sqrt(mean((final.test.predictions - y.test)^2)) else NA
  }

  # Return predictions, training accuracies, testing accuracy, combined accuracies, and variable importances
  return(list(
    train_predictions = final.train.predictions,
    test_predictions = if (!is.null(y.test)) final.test.predictions else NA,
    combined.train.metric = combined.train.accuracy,
    combined.test.metric = combined.test.accuracy
  ))
}
