#' Top K features
#'
#'
#' This function performs feature selection based on specified statistical methods. It supports three methods: correlation, chi-square, and PCA. The function is designed to handle different data types and select the most relevant features based on the method and the response variable characteristics.
#'
#'
#' @param X A matrix of predictor variables.
#' @param y A vector of the response variable.
#' @param k The number of features to select.
#' @param method The method used for feature selection, supported methods are "correlation" for continuous response, "chi-square" for categorical response, and "pca" for any type of response.
#'
#' @return A list containing two elements: `selected.indices` which are the indices of the selected features, and `selected.data` which is the subset of X corresponding to the selected features.
#' @export
#'
#' @examples
#' {
#'   Example using a hypothetical dataset
#'   X <- matrix(rnorm(100 * 10), ncol = 10)  # 100 observations and 10 features
#'   y <- rbinom(100, 1, 0.5)  # Binary outcome
#'   result <- pre.screen(X, y, k = 5, method = "correlation")
#'
#'   # Example using 'chi-square' method, requires y as factor
#'   y_factor <- factor(sample(c("Group1", "Group2"), 100, replace = TRUE))
#'   result_chi <- pre.screen(X, y_factor, k = 3, method = "chi-square")
#'
#'   # PCA method example
#'   result_pca <- pre.screen(X, y, k = 4, method = "pca")
#'}
#'
#' @seealso \code{\link[stats]{cor}}, \code{\link[stats]{chisq.test}}, \code{\link[stats]{prcomp}}
#'
#'
pre.screen <- function(X, y, k, method) {
  # Error Handling
  if (!is.matrix(X)) {
    stop("Error: X must be a matrix.")
  }
  if (!(is.vector(y) || is.null(y))) {
    stop("Error: y must be a vector or a factor.")
  }
  if (k > ncol(X)) {
    stop("Error: k cannot be greater than the number of columns in X.")
  }
  if (!(method %in% c("correlation", "chi-square", "pca", "randomForest"))) {
    stop("Error: Invalid method specified. Method should be from c('correlation', 'chi-square', 'pca', 'randomForest')")
  }

  unique_y <- unique(y)
  is_binary_response <- length(unique_y) == 2 && all(unique_y %in% c(0, 1))

  # Method-specific pre-processing and validation
  if (is_binary_response) {
    if (method == "correlation") {
      stop("Error: Correlation method cannot be used with a binary response.")
    }
    if (method == "chi-square" || method == "pca") {
      y <- as.numeric(as.factor(y))  # Convert binary response to numeric for chi-square
    }
  } else {
    if (method == "chi-square") {
      stop("Error: Chi-square method requires both predictor and response variables to be categorical/binary.")
    }
  }

  # Check if predictors are suitable for the method
  if (method == "correlation" && any(!sapply(X, is.numeric))) {
    stop("Error: Correlation method requires all predictors to be numeric.")
  }

  # Check if predictors are suitable for the method
  if (method == "chi-square" && any(!sapply(X, function(x) length(unique(x)) < 10 && all(x == as.integer(x))))) {
    stop("Error: Chi-square method requires all predictors to be categorical (ideally integer-coded).")
  }



  # Initialize variables
  indices <-  NULL

  # Selection Methods
  switch(method,
         "correlation" = {
           # Compute absolute correlations and order by their strength
           corr.scores <- abs(cor(X, y, use = "complete.obs"))
           indices <- order(corr.scores, decreasing = TRUE)[1:k]
         },

         "chi-square" = {
           if(!is.factor(y)) {
             stop("Error: Chi-square method requires 'y' to be a factor.")
           }
           # Perform chi-square tests for each feature
           # order by the chi-square statistic, descending
           chi.scores <- sapply(seq_len(ncol(X)), function(i) {
             res <- chisq.test(table(X[, i], y))
             res$statistic
           })
           indices <- order(chi.scores, decreasing = TRUE)[1:k]
         },
         "pca" = {
           # Perform PCA on scaled data
           pca.fit <- prcomp(X, scale. = TRUE)
           pca.loadings <- abs(pca.fit$rotation[, 1:k])

           # Calculate cumulative loadings across the first k components
           cumulative_loadings <- rowSums(pca.loadings)

           # Select the top k features based on cumulative loadings
           indices <- order(cumulative_loadings, decreasing = TRUE)[1:k]
         },
         "randomForest" = {
           if (is_binary_response){
             rf.model <- randomForest::randomForest(X, as.factor(y), importance = TRUE, ntree = 500)
           }
           else {
             rf.model <- randomForest::randomForest(X, y, importance = TRUE, ntree = 500, type = "regression")
           }
           importance.scores <- importance(rf.model, type = 1)
           indices <- order(importance.scores, decreasing = TRUE)[1:k]
         })


  if(!is.null(indices)) {
    data <- "No data selected. Wrong method"
  } else {
    data <- X[,indices, drop = FALSE]
  }

  # Return both indices and the actual data
  return(list(selected.indices = indices, selected.data = data))
}
