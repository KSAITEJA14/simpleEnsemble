#' Comprehensive Bagging Function for Various Regression and Classification Models
#'
#' This function implements a generic bagging algorithm which can be applied to different model types including linear, logistic, ridge, lasso, and elastic-net models. It handles feature importance and uses different methods for combining predictions.
#'
#'
#' @param X_train A matrix of predictor variables for training the models.
#' @param y_train A vector of the response variable used for training models.
#' @param X_test A matrix of predictor variables for testing the models.
#' @param y_test A vector of the response variable used for evaluating the models
#' @param model.type A character string specifying the type of model to be fitted. Supported model types include "linear", "logistic", "ridge", "lasso", "elastic", and "svm". This parameter determines the type of predictive model that will be applied to the input data. Each model type is handled according to its own specific implementation within the function.
#'
#' @param R The number of bootstrapping samples.
#'          This parameter is crucial when implementing methods that require
#'          repeated sampling to estimate model stability or performance.
#' @param alpha This parameter controls the mixing between lasso (L1) and ridge (L2) penalties.
#'          `alpha = 1` is lasso, while `alpha = 0` is ridge regression.
#'          Typically ranges between 0 and 1.
#' @param lambda The regularization parameter in ridge, lasso, and elastic net models.
#'          Controls the amount of shrinkage: the larger the lambda, the more shrinkage,
#'          thus driving coefficients to zero in lasso and towards zero in ridge.
#' @param intercept  logical value indicating whether the model should include an intercept term.
#'          Default is `TRUE`, which includes an intercept in the model. Set to `FALSE` to fit the model
#'          without an intercept, which may be appropriate in certain contexts, such as when data are centered.
#' @param kernel Specifies the type of kernel to be used in kernel-based learning algorithms, such as SVMs.
#'          Common choices include "linear", "polynomial", "radial", and "sigmoid". Each kernel type can
#'          influence the decision boundary and performance differently based on the nature of the data.
#' @param imp.threshold The importance threshold for feature selection.
#'          Features with importance measures (like variable importance in random forests) below this threshold
#'          may be excluded from the model. This helps in reducing model complexity and overfitting.
#' @param combine.method The method to be used for combining model predictions. Supported methods are "average" for simple averaging of predictions and "stacking" for combine predictions based on their performance using Meta learner which is an SVM.
#'
#' @return A vector of predictions resulting from the specified method of combining model outputs.
#'
#' @export
#'
#' @examples
#' #' {
#'   # Using the mtcars dataset for a logistic regression model example
#'   X <- as.matrix(mtcars[, 1:7])
#'   y <- as.integer(mtcars$am)  # am as a binary response variable
#'   result1 <- bag(X, y, model.type = "logistic", R = 10)
#'
#'   # Using the iris dataset for a linear regression model example
#'   X <- as.matrix(iris[, 1:4])
#'   y <- iris$Sepal.Length
#'   result2 <- bag(X, y, model.type = "linear", R = 10)
#'}
#'
#' @seealso \code{\link[stats]{lm}}, \code{\link[stats]{glm}}

bag <- function(X.train, y.train, X.test, y.test = NULL, model.type, R, alpha = NULL, lambda = NULL, intercept = TRUE, kernel = NULL, imp.threshold = NULL, combine.method = "stacking") {

  # Error Handling
  if (!(is.matrix(X.train) && is.matrix(X.test))) {
    stop("Error: X.train & X.test must be matrices.")
  }
  if (!(ncol(X.train) == ncol(X.test))) {
    stop("Error: X.train and X.test must have same number of features")
  }
  if (!is.vector(y.train)) {
    stop("Error: y.train must be a vector.")
  }
  if (!is.null(y.test) && !is.vector(y.test)) {
    stop("Error: y.test must be a vector.")
  }
  if (!(model.type %in% c("linear", "logistic", "ridge", "lasso", "elastic", "svm"))) {
    stop("Error: Invalid model_type.")
  }
  if (!is.null(alpha) && (alpha < 0 || alpha > 1)) {
    stop("Error: alpha should be between 0 and 1.")
  }
  if (is.null(imp.threshold)) imp.threshold <- 1e-4

  X.train = na.omit(X.train)
  X.test = na.omit(X.test)

  # Initialize matrices for predictions and importances
  prediction.matrix <- matrix(NA, nrow = nrow(X.test), ncol = R)
  importance.matrix <- matrix(0, nrow = ncol(X.train), ncol = R)
  training_predictions <- matrix(NA, nrow = nrow(X.train), ncol = R)

  # Check response type for binary classification
  unique_y <- unique(y.train)
  is_binary_response <- length(unique_y) == 2 && all(unique_y %in% c(0, 1))
  if (is_binary_response) {
    if (!(combine.method %in% c("majority", "stacking"))) {
      stop("Method should be 'majority' or 'stacking' when response variable is binary")
    }
  } else {
    if (!(combine.method %in% c("average", "stacking"))) {
      stop("Method should be 'average' or 'stacking' when response variable is continuous")
    }
  }

  # Model fitting, predictions, and importance calculations
  for (i in 1:R) {
    indices <- sample(1:nrow(X.train), replace = TRUE)
    X_sample <- X.train[indices, ]
    y_sample <- y.train[indices]
    bag.model <- fit.model(X = X_sample, y = y_sample, model.type = model.type, alpha = alpha, lambda = lambda, intercept = intercept, kernel = kernel)

    # Store predictions for train and test
    training_predictions[, i] <- predict(bag.model, X.train)
    prediction.matrix[, i] <- predict(bag.model, X.test)

    # Calculate and store importances
    if (model.type %in% c("ridge", "lasso", "elastic")) {
      importance.matrix[, i] <- abs(coef(bag.model)[-1]) > imp.threshold
    } else {
      importance.matrix[, i] <- coef(bag.model)[-1] != 0
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
  final.train.predictions <- combine_predictions(training_predictions, y.train, combine.method, is_binary_response)
  final.test.predictions <- combine_predictions(prediction.matrix, y.test, combine.method, is_binary_response)

  if (is_binary_response) {
    combined.train.accuracy <- mean(final.train.predictions == y.train)
    combined.test.accuracy <- if (!is.null(y.test)) mean(final.test.predictions == y.test) else NA
  } else { # Calculate combined RMSE for continuous response
    combined.train.accuracy <- sqrt(mean((final.train.predictions - y.train)^2))
    combined.test.accuracy <- if (!is.null(y.test)) sqrt(mean((final.test.predictions - y.test)^2)) else NA
  }

  # Compute final importance scores as the average over all bootstraps
  final.importances <- rowSums(importance.matrix)

  # Return predictions, training accuracies, testing accuracy, combined accuracies, and variable importances
  return(list(
    train_predictions = final.train.predictions,
    test_predictions = if (!is.null(y.test)) final.test.predictions else NA,
    combined.train.metric = combined.train.accuracy,
    combined.test.metric = combined.test.accuracy,
    importances = final.importances
  ))
}

