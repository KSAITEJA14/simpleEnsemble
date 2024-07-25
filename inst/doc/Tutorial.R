## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE, error = TRUE)
library(SimpleEnsemble)

## ----testing-fit.model--------------------------------------------------------
X <- as.matrix(mtcars[, 2:8])
y <- mtcars$mpg
print(y)

model.ridge <- fit.model(X, y, model.type = "ridge")
coef(model.ridge)

model.lasso <- fit.model(X, y, model.type = "lasso", lambda = 0.002)
coef(model.lasso)
 
model.linear <- fit.model(X, y, model.type = "linear")
coef(model.linear)

model.elastic <- fit.model(X, y, model.type = "elastic")
coef(model.elastic)

model.svm <- fit.model(X, y, model.type = "svm")
#you can use this model object of SVM type to predict.

## ----fit.model - example classification models--------------------------------
X <- as.matrix(mtcars[, 1:7])
y <- as.integer(mtcars$am)
print(y)

model.logistic <- fit.model(X, y, model.type = "logistic")
coef(model.logistic)

model.ridge <- fit.model(X, y, model.type = "ridge")
coef(model.ridge)

model.lasso <- fit.model(X, y, model.type = "lasso")
coef(model.lasso)

model.elastic <- fit.model(X, y, model.type = "elastic")
coef(model.elastic)

model.svm <- fit.model(X, y, model.type = "svm")
#you can use this model object of SVM type to predict.

## ----Bagging - classification:defauit (stacking)------------------------------

data <- read.csv("H:/My Drive/Spring24/AMS597/SimpleEnsembleGroup21/SimpleEnsembleGroup21/data/Enigma.csv", header = TRUE)

# Define X and y
X <- data[,1:13]
y <- as.vector(data$y)

# Split the data into training and testing sets
set.seed(123) # for reproducibility

# Create the training set indices using createDataPartition
library(caret)
training_samples <- createDataPartition(y, p=0.75, list=FALSE)

# Define training and testing sets
X_train <- as.matrix(X[training_samples, ])
y_train <- y[training_samples]
X_test <- as.matrix(X[-training_samples, ])
y_test <- y[-training_samples]
result1 <- SimpleEnsemble::bag(X_train, y_train, X_test,y_test, model.type = "lasso", R = 10)
print(paste("train accuracy:", result1$combined.train.metric))
print(paste("test accuracy:", result1$combined.test.metric))
print(result1$importances)

## ----Bagging - classification:majority----------------------------------------
result1 <- SimpleEnsemble::bag(X_train, y_train, X_test,y_test, model.type = "lasso", R = 10, combine.method = "majority")
print(paste("train accuracy:", result1$combined.train.metric))
print(paste("test accuracy:", result1$combined.test.metric))
print(result1$importances)

## ----Bagging - regression:defauit (stacking)----------------------------------
# Load the data
data <- read.csv("H:/My Drive/Spring24/AMS597/SimpleEnsembleGroup21/SimpleEnsembleGroup21/data/QuestionMark.csv", header = TRUE)


# Define X and y
X <- data[, -which(names(data) == "w4")]
y <- as.vector(data$y)

# Split the data into training and testing sets
set.seed(123) # for reproducibility
training_samples <- createDataPartition(y, p=0.75, list=FALSE)

# Define training and testing sets
X_train <- as.matrix(X[training_samples, ])
y_train <- y[training_samples]
X_test <- as.matrix(X[-training_samples, ])
y_test <- y[-training_samples]


result1 <- SimpleEnsemble::bag(X_train, y_train, X_test,y_test, model.type = "elastic", R = 10)
print(paste("train RMSE:", result1$combined.train.metric))
print(paste("test RMSE:", result1$combined.test.metric))
print(result1$importances)

## ----Bagging:average----------------------------------------------------------
result1 <- SimpleEnsemble::bag(X_train, y_train, X_test,y_test, model.type = "elastic", R = 10, combine.method = "average")
print(paste("train RMSE:", result1$combined.train.metric))
print(paste("train RMSE:", result1$combined.test.metric))
print(result1$importances)

## ----top k: data--------------------------------------------------------------
data <- read.delim("https://www.ams.sunysb.edu/~pfkuan/Teaching/AMS597/Data/leukemiaDataSet.txt", header=TRUE, sep='\t')
data$Group <- ifelse(data$Group == "ALL", 1, 0)

## ----top k:continuous X, continuous y-----------------------------------------
y <- data$Gene2238
X <- as.matrix(data[, -c(1,2238)])
prescreen.result.corr <- pre.screen(X, y, 10, method = "correlation")
print("correlation")
print(prescreen.result.corr$selected.indices)

prescreen.result.pca <- pre.screen(X, y, 10, method = "pca")
print("pca")
print(prescreen.result.pca$selected.indices)

prescreen.result.rf <- pre.screen(X, y, 10, method = "randomForest")
print("random forest")
print(prescreen.result.rf$selected.indices)

## -----------------------------------------------------------------------------
y <- data$Group
X <- as.matrix(data[, -c(1)])
prescreen.result.rf.cont <- pre.screen(X, y, 10, method = "randomForest")
print("random forest")
print(prescreen.result.rf.cont$selected.indices)

## -----------------------------------------------------------------------------
data <- read.delim("https://www.ams.sunysb.edu/~pfkuan/Teaching/AMS597/Data/leukemiaDataSet.txt", header=TRUE, sep='\t')
data$Group <- ifelse(data$Group == "ALL", 1, 0)
y <- as.vector(data$Gene1)
X <- as.matrix(data[, -c(1,2)])
prescreen.result <- pre.screen(X, y, 10, method = "chi-square")


## -----------------------------------------------------------------------------
# Load the data
data <- read.csv("H:/My Drive/Spring24/AMS597/SimpleEnsembleGroup21/SimpleEnsembleGroup21/data/Enigma.csv", header = TRUE)

# Define X and y
X <- data[,1:13]
y <- as.vector(data$y)

# Split the data into training and testing sets
set.seed(123) # for reproducibility

# Create the training set indices using createDataPartition
library(caret)
training_samples <- createDataPartition(y, p=0.75, list=FALSE)

# Define training and testing sets
X_train <- as.matrix(X[training_samples, ])
y_train <- y[training_samples]
X_test <- as.matrix(X[-training_samples, ])
y_test <- y[-training_samples]
 
model.specs <- list(
  list(model="ridge", lambda=0.1),
  list(model="lasso", lambda=0.1),
  list(model="svm", kernel="radial")
)

predictions <- SimpleEnsemble::ensemble(X_train, y_train, X_test, y_test, model.specs)
print(paste("train accuracy:", predictions$combined.train.metric))
print(paste("test accuracy:", predictions$combined.test.metric))

## -----------------------------------------------------------------------------
predictions <- SimpleEnsemble::ensemble(X_train, y_train, X_test, y_test, model.specs, combine.method = "majority")
print(paste("train accuracy:", predictions$combined.train.metric))
print(paste("test accuracy:", predictions$combined.test.metric))

## -----------------------------------------------------------------------------
# Load the data
data <- read.csv("H:/My Drive/Spring24/AMS597/SimpleEnsembleGroup21/SimpleEnsembleGroup21/data/QuestionMark.csv", header = TRUE)


# Define X and y
X <- data[, -which(names(data) == "w4")]
y <- as.vector(data$y)

# Split the data into training and testing sets
set.seed(123) # for reproducibility
training_samples <- createDataPartition(y, p=0.75, list=FALSE)

# Define training and testing sets
X_train <- as.matrix(X[training_samples, ])
y_train <- y[training_samples]
X_test <- as.matrix(X[-training_samples, ])
y_test <- y[-training_samples]

model.specs <- list(
  list(model="ridge", lambda=0.1),
  list(model="lasso", lambda=0.1),
  list(model="svm", kernel="radial")
)

result1 <- SimpleEnsemble::ensemble(X_train, y_train, X_test, y_test, model.specs)
print(paste("train RMSE:", result1$combined.train.metric))
print(paste("test RMSE:", result1$combined.test.metric))

## -----------------------------------------------------------------------------
result1 <- SimpleEnsemble::ensemble(X_train, y_train, X_test, y_test, model.specs, combine.method = "average")
print(paste("train RMSE:", result1$combined.train.metric))
print(paste("test RMSE:", result1$combined.test.metric))

