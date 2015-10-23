### Predictive Machine learning John Hopkins uni

## Lecture 1
## Lecture 1.1)
# study design: training vs test sets
# conceptual issues: out of sample error, ROC curves

## Lecture 1.2)
# question -> input data -> features -> algorithm -> parameters -> evaluation

## Lecture 1.3) 
# Garbage in = Garbage out
# 1) May be easy (movie ratings -> new movie ratings)
# 2) May be harder (gene expression data -> disease)
# 3) Depends on what is a good prediction
# 4) Often more data -> better models
# 5) Most important steps

# Features
# Lead to data compression
# Retain relevant information
# Created based on expert application knowledge

## Prediction is about accuracy tradeoffs
# intepretability versus accuracy
# simplicity vs accuracy

## 1.4) In sample and out of sample errors
# in sample error: error you get on same data set you used to build your predictor
# sometimes called resubstitution error

# out of sample error: The error you get on a new data set. Sometimes called generalization error

## Key ideeas
# out of sample error is what you care about
# in sample error < out sample error
# reason is overfitting. You are matching you algorithm to the data you have

## Every dataset we have signal component and noise
# goal of predictor is to find signal, and ignore the noise
# you can always design perfect in sample predictor
# You capture both signal + noise when you do that

## 1.5) Prediction study design
# 1) Define your error rate
# 2) Split data into, Training, testing, validation
# 3) On training set pick features
# 4) ON training set pikc prediction function, use cross-validatino
# 5) If no validation, apply 1x to test set
# 6) IF validation, appy to test set and refine. Apply 1x to validation

# RUles of thumb for prediction study design
# 60% training
# 20% test
# 20 % validation

## 1.6) Types of errors
# positive = identified
# negative = rejected
# sensitivity: TP / (TP + FN)
# specificity: TN / (FP + TN)

# 99 % sensitivity and specificity
# Imagine 90.000 people
# Positive Predictive Value: 9900 / (9900 + 900) = 92% 
# Negative Predicive value: 89100 / (100 + 89100) = 99.9% 
# Accuracy: (9900 + 89100) / 10000 = 99 %

# Mean squared error (MSE) = 1 / n * SUM(Prediction - Truth) ^2
# Root mean squared error (RMSE) = sqrt(1 / n * SUM (Prediction - Truth) ^2)
# Sensitive to outliers

## Common error measures
# Median absolute deviation
# Sensitivity: If you want few missed positives
# Specificity: IF you want few negatives called positives
# Accuracy: Weights false positives / negatives equally
# Concodrance: (e.g. kappa)

## 1.7) ROC cuve
# Predictions quantitative
# proabilities
# x axis = 1 - P(FP) , y = P(TP)

# Area under the curve (AUC)
# AUC = 0.5 is random quessing
# AUC of above 0.8 is considered good
# the further you are to the top left corner, better ROC curve

## 1.8) Cross validation
# Accuracy on trainig set (resubstition accuracy) is optimistic
# Subsplit data in training and test sets

# K-fold cross validation
# apply model to different parts of data and test on different parts of data

# Leave one out cross-validation

# For k-fold cross validation
# Large k = less bias, more variance
# smaller k = more bias, less variance

# Random sampling with replacement is bootstra
# underestimates of error

## 1.9) What data should you use?
# data that is as closely related to what you want to predict


### Quizes

# Question 1) Steps in building machien learning algorithm

# Question 2) ALgorith 100% accuracy training data. Why not work well on other data?
# overfitting

# Question 3) Sizes training and test data
# 60% training, 40% testing

# Question 4)

### Week 2

## 2.1) Caret package
# Why caret?
# use just 1 function
library(caret)
library(kernlab)
# use 75% of data to train the model
inTrain <- createDataPartition(y = spam$type,
                               p = 0.75, list = FALSE)
# subsetting the data
training <- spam[inTrain,]
testing <- spam[-inTrain,]
dim(training)

# check out the final model
modelFit <- train(type ~., data = trainng, method = "glm")
modelFit$finalModel

# now predict it using our algorithm. Use the testing data for new data
predictions <- predict(modelFit, newdata = testing)

# confusionMatrix
# A table with accuracy, sensitivity and specificity
confusionMatrix(predictions, testing$type)

# Tutorials to check out
http://www.edii.uclm.es/~useR-2013/Tutorials/kuhn/user_caret_2up.pdf
http://cran.r-project.org/web/packages/caret/vignettes/caret.pdf

## 2.2) Data slicing
# create K folds
library(caret); set.seed(32323); data(spam)
training <- spam[inTrain,]
testing <- spam[inTrain,]

# Make smaller folds
folds <- createFolds(y = spam$type, k = 10, list = TRUE, returnTrain = TRUE)
sapply(folds, length)

# returnTrain = FALSE. THen it will return testset
# you can also de resample
set.seed(32323)
folds <- createResample(y = spam$type, times = 10,
                        list = TRUE)

# create time slices
set.seed(32323)
tm <- 1:1000
folds <- createTimeSlices(y = tme, initialWindow = 20,
                          horizon = 10)

## 2.3) Training control
intrain <- createDataPartition(y = spam$type, 
                               p = 0.75)
# Metric options
# Continuous outcomes
# RMSE = Root mean squared error
# RSquared = R^2 from regression models
## Categorical outcomes
# Accuracy: fraction of correct

## trainControl

### Tutorial caret package

# test data set
data(segmentationData)
# get rid of the cell identifier
segmentationData$Cell <- NULL
training <- subset(segmentationData, Case == "Train")
testing <- subset(segmentationData, Case == "Test")
training$Case <- NULL
testing$Case <- NULL
str(training[,1:6])

## Common steps during model building are:
# estimating model parameters
# determining the values of tuning parameters
# calculating the performance of the final model

## Preporcessing predictors
trainX <- training[, names(training) != "Class"]
preProcValues <- preProcess(trainX, method = c("center", "scale"))


