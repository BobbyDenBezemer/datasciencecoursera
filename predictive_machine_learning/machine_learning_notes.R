
# Arthur Samuel (1959)
# computers ability to learn without being programmed to
# TOm Mitchel
# A computer program is said to learn from experience E
# with some task T and some performance measure P, if its performance
# on T, as measured by M, is said to improve with E.

## Supervised learning
# Example housing price prediction
# price versus square feet house size
# give the algorithm a dataset where the right answers are given.
# task of the algorithm was coming up with new right answer for new problems (new houses)
# Regression problem.
# Other example: Classification problem (zero or 1) malignant or benign tumor


## Unsupervised learning
# data has no labels and we are not given what to do with the data
# Can you find some structure in the data set?
# clustering algorithm
# cocktail party problem

## First learning algorithm: Linear regression
# supervised learning: Given the right answer for each example in the data
# regressino problem: Predict real valued output
# m = number of training examples
# x's = input variable / features
# y's = output variable / target
# (x, y) = single row (training example)
# specifc training example (xi, yi)
# size of house -> hypothesis -> Estimated price
# univariate linear regression

# Cost function: how to fit the best possible line through our data
# htheta(x) = theta0 + theta1 * x
# theta are parameters in the model
# minimize theta0 and theta1
# sum(h0(xi) - yi) ^2 . Minimize difference predicted and actual score
# htheta(xi) = theta0 + theta1x(i)
# squared error cost function: J(theta0, theta1) = 1 / 2*m * sum(h0(xi) - yi) ^2

## Example
# say you set theta1 = 2 and you have a dataset with 3 values.
# (1,1), (2,2) (3,3), the cost function J(2) then is:
# 1/ 2 * 3 * sum((1 -2)^2 + (4-2)^2 + (6 - 3) ^2 ) = 2.33

# Gradient descent
# start with some Theta0 and Theta1
# Keep changing theta0 and theta1 reduce J(theta0,theta1)
# until we hopefully end up at a minimum
# sometimes you can however end up at different local optima

# repeat until convergence
# thetaj := thetaj - a * derivative term
# a = learning rate (how big the step down the hill is)
# simultaneously update theta0 and theta1
# Correct steps
# temp0 := theta0 - a * derivative J(theta0,theta1)
# temp1 := theta1 - a * derivative J(theta0,theta1)
# theta0 := temp0; theta1 := temp1

# derivative takes tangent of a certan point and calculates the slope at the given point.
# if the derivative is positive, you'll update theta 1 with - alpha.
# This is how you can minimize J
# If yourlearning rate, alpha is too small, it takes a lot of time to get to the minimum
# If your learning rate, alpha is too large, gradient descent overshoot minimum.
# if your theta1 is already at the local optimum, the derivative term is zero at that point
# in your gradient descent update doesnt happen, a* 0 = 0. It fails to go to the global optimum
# as we approach a local minimum, gradient descent will automatically take smaller steps
# because the derivative gets smaller.

# combining both
# theta0 = theta0 - a * 1 / m* sum(htheta(xi) - yi)
# theta1 = theta1 - a * 1 / m * sum(htheta(xi) -y(i)) * xi
# Batch Gradient Descent ->each step of gradient descent uses all the training examples

## Linear algebra review
# What are matrices and vectors
# matrix has dimensions: rows and columns
# vector is matrix with only 1 column

# scalar (number)multiplicaton
# 3 * matrix(c(1,2,3,4,5,6),nrow = 3,byrow = TRUE) = matrix(c(3,6,9,12,15,18),nrow = 3,byrow = TRUE)
# dimensionality stays the same

# matrix vector multiplication. Number of colums matrix 1 should be equal to number of rows matrix 2
# multiply row 1 of matrix1 with column 1 of matrix2 -> This gives first element of new matrix.
# result of 3 * 4 matrix * vector with 4 columns -> vector with 3 rows and 1 column

# Using matrix multiplication for regression
# htheta(x) = -40 + 0.25*x
# matrix(c(1,1,1,1,2104,1416,1534,852),byrow = FALSE,nrow = 4) * v(-40,0.25) ->

# matrix matrix multiplication
# row1 * column1, row2 * column1, row1 * column 2, row2 * column 2
# 2*3 matrix * 3*2 matrix = 2 *2 matrix
# matrix(1,2,3,5,byrow = FALSE) * matrix(0,3,1,2,byrow = FALSE) = matrix((1*0+3*2),(2*0 + 5 *3),(1*1 + 3*2),(2*1 +5*2)

# matrix multiplication properties
# A * B != B a A (not commutative)
# associative -> 3*5*2 = 3*(5 * 2); (a *b) *b = a  (b* c)
# Identity Matrix = 1 on the diagonal, 0 everywhere else
# A * I = I * A = A

# Inverse and Transpose
# 3 inverse = 1/3, 12
# If A is an m * m matrix, and if it has an inverse
# A(A^-1) = A^-1 * A = I; Only square matrices have inverses
# solve(A) to get inverse in R
# Matrix Transpose is that you reverse rows and columns
# A = 1,2,0,3,5,9,nrow =2, byrow = TRUE -> 1,2,0,3,5,9,nrow = 3, byrow = FALSE

### WEEK 2

## Multiple features
# now we are using 4 features, so n = 4
# m = number of rows = 47. 
# x(i) = vector of ith training example = c(1416,3,2,40)
# hypothesis: Htheta(x) = theta0 + theta1x1 + theta2x2 etc.
# for convenience x0 = 1
# rowvector of thetas -> theta0, theta1, thetan
# column vector of xs => x0, x1, xn
# multivariate regression

# Feature Scaling
# Idea: Make sure features are on the same scale
# x1 = size of house / 2000, x2 = number of bedrooms / 5
# helps your algorithm to converge faster
# So get xs between -1 and 1.
# Mean normalization
# x1 = size - 1000 / 2000; x2 = number bedrooms - 2 / 5
# x1 - averagex / SD

# Learning rate
# J(theta) should decrease after every iteration.
# declare convergence if J(theta) decreases by less than 10^-3 in one iteration
# gradient descent not working, use smaller alpha
# if a (learning rate) is too small: very slow convergence rate

# housing prices prediction
# polynomial regression
# theta0 + x*theta1 + theta2*x^2 -> quadratic model
# theta0 + x*theta1 + theta2*x^2 + theta3*x^3 -> cubic
# theta0 + theta1 * size + theta2 * sqrt(size)

### Predictive machine learning week 2
### Caret package
# WHy caret?
lda -> Mass package
glm -> stats package
gbm -> gbm package
mda -> mda package
rpart -> rpart package
Weka -> RWeka package
LogitBoost -> caTools

# Quick example
library(caret); library(kernlab); data(spam)
inTrain <-createDataPartition(y=spam$type,
                              p = 0.75, list = FALSE)
training <- spam[inTrain,]
testing <- spam[inTrain,]

# now fit a model
set.seed(32343)
modelFit <- train(type ~., data=training, method="glm")

# creates a model fit for training model
# look at final model
modelFit$finalModel
# predict on new samples
predictions <- predict(modelFit, newdata = testing)
# confusion matrix calculation
# gives you accuracy etc and sensitivity, specificity
confusionMatrix(predictions, testing$type)

# further documentation
http://www.jstatsoft.org/v28/i05/paper
http://www.edii.uclm.es/~useR-2013/Tutorials/kuhn/user_caret_2up.pdf
http://cran.r-project.org/web/packages/caret/vignettes/caret.pdf

## Data slicing
library(caret); library(kernlab); data(spam)
inTrain <- createDataPartition(y=spam$type,
                               p=0.75, list=FALSE)
# Subsetting the data
training <- spam[inTrain,]
tresting <- spam[-inTrain,]
dim(training)

# Train a model
set.seed(32323)
modelFit <- train(type~., data = training, method = "glm")
modelFit$finalModel

# You can predict on new samples using the predict command
predictions <- predict(modelFit, newdata = training)
confusionMatrix(predictions, testing$type)

### Creating data partitions
# Make k-fold  sets
set.seed(32323)
folds <- createFolds(y=spam$type,k=10,
                     list=TRUE, returnTrain = TRUE)
sapply(folds,length)

# Resampling
set.seed(32323)
folds <- createResample(y=spam$type, times=10,
                        list=TRUE)
# Time slices
folds <- createTimeSlices(y=tme, initialWindow = 20,
                          horizon=10)
names(folds)
# further information, see above

## Training options
library(caret); library(kernlab); data(spam)
inTrain <- createDataPartition(y = spam$type,
                               p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
# training options
args(train.default)

## Metric options
## continous outcomes
# RMSE = root mean squared error
# RSquared = from regression models
## Categorical outcomes
# accuracy = fraction correct
# kappa

# Options for resampling
# boot = bootstrapping, boot632 = bootstrapping with adustment, cv = cross validation

## Setting the seed
# oftne useful to set an overall seed otherwise you get a different answer everyt time you run ur code
set.seed(1235)
modelFit2 <- train(type ~, data=training, method="glm")
# further reading
http://caret.r-forge.r-project.org/training.html

### Plotting predictors
library(ISLR); library(ggplot2); library(caret)
data(Wage)

# Get training/test sets
inTrain <- createDataPartition(y = Wage$wage,
                               p=0.7, list=FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
dim(training); dim(testing)
# do the plotting
featurePlot(x = training[,c("age","education","jobclass")],
            y = training$wage, plot = "pairs")


# Breaking up variables into different categories
library(Hmisc)
cutWage <- cut2(training$wage, g=3)
table(cutWage)

# Boxplot
qplot(cutWage, age, data = training, fill = cutWage, geom = "boxplot")

# Tables
t1 <- table(cutWage, training$jobclass)
prop.table(t1, 1) # get a proportional table

# Density plot
qplot(wage, colour = education, data = training, geom="density")

# Notes and further reading
# Look for: Imbalance in outcomes/predictors
# outliers, group of points not explained by predictors
# skewed variables
http://topepo.github.io/caret/visualizations.html

### Basic preprocessing
## Why preprocess?
# IF variables are skewed -> STandardizing
# Note we have to use the mean and sd from test set on training set
set.seed(32343)

# You can also pass the preProcess function to train
inTrain <- createDataPartition(y = spam$type, p = 0.75, list = FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
modelFit <- train(type ~., data = training, 
                  preProcess = c("center","scale"), method = "glm")

# Standardizing - Box-Cox
preObj <- preProcess(training[,-58], method = c("BoxCox"))

# Standardizing - Imputing
set.seed(13343)

# Make some values NA
training$capAve <- training$capitalAve
selectNA <- rbinom(dim(training)[1], size = 1, prob = 0.05)==1
training$capAve[selectNA] <- NA

# impute and standardize
preObj <- preProcess(training[,-58], method = "knnImpute")
capAve <- predict(preObj, training[,-58])$capAve

# standardize true values
capAveTruth <- training$capitalAve
capAveTruth <- (capAveTruth - mean(capAveTruth)) / sd(capAveTruth)

# compare actual value and imputed values
quantile(capAve - capAveTruth)

# Notes and further reading
# training and test sets musit be processed in the same way
http://topepo.github.io/caret/preprocess.html

## Covariate creation
## Level 1: 
# from raw data to covariate
# text files: frequency of words, frequency of phrases, average capital letters
# images: Edges, corners, blobs, ridges
# webpages: number and types of images, position of elements
# people: height, weight, hair color, sex

## Level 2: Tidy covariates
# More necessary for some methods (regression, svm) than others (classification trees)
# should be done only on the training set
# best approach is through exploratory analyses
# Balancing summarization vs information loss
# when in dobut, create more features

## Example
library(caret); library(ISLR); data(Wage)
inTrain <- createDataPartition(y = Wage$wage,
                               p=0.7, list=FALSE)

training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
# Basic idea is to convert factor variables to dummy variables
table(training$jobclass)

# Basic idea, convert factor variables to indicator variables
dummies <- dummyVars(wage ~jobclass, data = training)
head(predict(dummies, newdata = training))

## Removing zero covariates
# Some variables have zero variability
nsv <- nearZeroVar(training, saveMetrics = TRUE)
# look at percentUnique and nzv columns
# sex is only basically male, and low frequency ratio, so it's near zero

## Spline basis
# Sometimes you want to be able to fit curvy lines instead of straight lines
library(splines)
bsBasis <- bs(training$age, df=3) # third degree polynomial
bsBasis # 1st column: age 2nd column: 2nd degree polynimal, 3rd column: age cube
# fitting curves with spline
lm1 <- lm(wage ~bsBasis, data = training)
plot(training$age, training$wage, pch = 19)

# On test dataset you have to use exact same procedure as on the training set
predict(bsBasis, age = testing$age)

## Notes and further reading
# google feature extraction for type of data ur analyzing
http://www.cs.nyu.edu/~yann/talks/lecun-ranzato-icml2013.pdf
# create new covariates if you think they will improve fit
http://datasciencespecialization.github.io/courses/08_PracticalMachineLearning/015covariateCreation/#11
  
## Preprocessing with principal component analysis
# sometimes very highly correlated predictors
# good to make a summary variable
library(caret); library(kernlab); data(spam)
inTrain <- createDataPartition(y=spam$type, 
                               =0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
M<- abs(cor(training[,-58]))
diag(M) <- 0
which(M > 0.8, arr.ind=T)
# columns 32 and 34
plot(spam[,34],spam[,32])

## Basic PCA idea
# a weighted combination of predictors explains most of what is going on
# reduce number of predictors
# reduce noise (due to averaging)

# We could rotate the plot 
## Related problems
# you have multivariate variables X1, Xn
# find a new set of multivariate variables that are uncorrelated
# and explain as much variability as possible
# if you put all the variables together in one matrix, find the best created matrix with


## Goals
# 1. Find a new set of multivariate variables that are uncorrelated and explain as much
# variance as possible
# 2. If you put all the variables together in one matrix, find the best matrix created with fewer
# variables that explains the original data


# SVD
# if x is a matrix with each variable in a column and each observation is a row
# SVD is a matrix decomposition
# where columns of U are orthogonal

# Example
smallSpam <- spam[,c(34,32)]
prComp <- prcomp(smallSpam)
plot(prComp$x[,1], prComp$x[,2])
prComp$rotation
# numb415 and PC1 = 0.7881 and PC@ = 0.7061
# num857 0.7061 

## PCA on SPAM data
typeColor <- ((spam$type=="spam") *1 + 1 )
prComp <- prcomp(log10(spam[,-58] + 1))
plot(prComp$x[,1],prComp$x[,2],col=typeColor,xlab ="PC1",ylab="PC2")

## PCA with caret
preProc <- preProcess(log10(spam[,-58] + 1), method="pca",pcaComp = 2)
spamPC <- predict(preProc, log1-(spam[,-58] + 1))
modelFit <- train(training$type ~. , method = "glm", data = trainPc)

# Alternative
modelFit <- train(training$type ~., method = "glm", preProcess = "pca", data = training)
confusionMatrix(testing$type, predict(modelFit,))

## FInal thoughts on PCS
# most useful for linear-type models
#can make it harder to interpret predictors
# Watch out for outliers
# transform first 
# plot predictors to identify problems

## Predicting with regression
# fit a simple regression model
# plug in new covariates and multiply by the coefficients
# useful when the linear model is nearly correct

# Example old faithful
library(caret); data(faithful); set.seed(333)
inTrain <- createDataPartition(y=faithful$waiting, p=0.5, list=FALSE)
trainFaith <- faithful[inTrain,]; testFaith <- faithful[-inTrain,]

# Fit a linear model
#X ED1 = b0 + b1x1 +e
library(ggplot2)
lm1 <- lm(eruptions ~ waiting, data = trainFaith)
ggplot(data = eruptions, aes(x = waiting, y = eruptions)) +
  geom_point() +
  geom_smooth(method = "lm")
# Predict a new value
ED = b0 +b1X1 # here we dont have an error terms as we dont know it
coef(lm)[1] + coef(lm1)[2] * 80
newdata <- data.frame(waiting = 80)
predict(lm1, newdata)

# See how the training set performs on the test set
par(mfrow=c(1,2))
plot(trainFaith$waiting, trainFaith$eruptions, pch= 19,col="blue")
lines(trainFaith$waiting,predict(lm1),lwd=3)
plot(testFaith$waiting,testFaith$eruptions,pch=19)
lines(testFaith$waiting,predict(lm1,newdata=testFaith), lwd =3)

# Get training set/test set errors
# calculate RMSE on training
sqrt(sum((lm1$fitted-trainFaith$eruptions) ^2))
# calculate RMSE on test
sqrt(sum((lm1$fitted-testFaith$eruptions) ^2))

# Prediction intervals
pred1 <- predict(lm1, newdata = testFaith, interval ="prediction")
ord <- order(testFaith$waiting)
plot(testFaith$waiting, testFaith$eruptions, pch = 19, col = "blue")
matlines(testFaith$waiting[ord], pred1[ord,], type = "l", col = c(1,2,2), lty = c(1,1,1))

# Same process with caret package
modFit <- train(eruptions ~ waiting, data = trainFaith)
summary(modFit$finalModel)

## Predicting with regression, multiple covariates
# Ge training/test sets
library(ISLR)
data(Wage)
Wage <- subset(Wage, select = -c(logwage))
summary(Wage)
inTrain <- createDataPartition(y = Wage$wage, p = 0.7, list = FALSE)

featurePlot(x = inTrain[,c('age', 'education', 'jobclass')],
                         y = training$wage,
                         plot = "pairs")
# Plot age versus wage
qplot(age, wage, data = training)

# Plot age versus wage colour by education
qplot(age, wage, colour = education, data = training)

# Fit a linear model
# ED = b0 + b1*age + b2*Jobclass
modFit <- train(wage ~ age+ joblcass + education,
                method = "lm", data = training)
finMod <- modFit$finalModel

# Diagnostics
plot(finMod, 1, pch = 19, cex = 0.5, col = "red")

# If you want all covariates
modFitAll <- train(wage ~. , data= training, method = "lm")
pred <- predict(modFitAll,)

### Quiz 2
library(AppliedPredictiveModeling)
library(caret)
data(AlzheimerDisease)
# make frames
adData = data.frame(diagnosis,predictors)
testIndex = createDataPartition(diagnosis, p = 0.50,list=FALSE)
training = adData[-testIndex,]
testing = adData[testIndex,]
dim(testing)

## Question 2
library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]
ggplot(data = training, aes(x = Superplasticizer)) +
  geom_bar()

ggplot(data = training, aes(x = log(Superplasticizer))) +
  geom_bar()

# Q3
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

library(dplyr)
training <- select(training, starts_with("IL"), diagnosis)
testing <- select(testing, starts_with("IL"), diagnosis)
preProc <- preProcess(training[,-13], method="pca", thresh = 0.8)

## Q4
#NO PCA Model
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
training <- select(training, starts_with("IL"), diagnosis)
testing <- select(testing, starts_with("IL"), diagnosis)
modFitnoPCA <- train(training$diagnosis ~., method="glm", data=training)
confusionMatrix(testing$diagnosis, predict(modFitnoPCA, testing))

#PCA Model
preProc <- preProcess(training[,-13], method="pca", thresh = 0.8)
trainPc <- predict(preProc, training[,-13])
modelFit2 <- train(training$diagnosis~.,data = trainPc, method = "glm")

testPc <- predict(preProc, testing[,-13])
confusionMatrix(testing$diagnosis, predict(modelFit2, testPc))


### Week 3
## Decision trees
# iteratively split variables into groups
# evaluate homogeneity within each group
# split again if necessary, until groups are homogenous or small enough

## Pro's
# easy to intrepret
# better performance in nonlinear settings

## Con
# without pruning / corss validation lead to overfitting
# harder to estimate uncertainty
# results may be variable

## Basic algorithm
# 1)start with all variables in one group
# 2) find variable /split that best seperates the 
# 3) divide the data into two groups
# 4) within each split, find the best variable/split
# 5) continue until groups are too small / suffiicently homogenous

## Measures of impurity
# Pmk = 1/ Nm number of times that k occurs 
# 1 - pmk misclassification error
# 0 = perfect purity, 0.5 = no purity
## Gini index
# 1 - squared probabilities that you belong to one of the given classes
# 0 = perfect purity; 0.5 = no purity

## Deviance / information gain
# probabilty ur assigned to group k * logpm
# 0 = perfect purity, 1 = no purity

## Example
# 15 blue dots, 1 red that should be blue
# misclassification of 1/16
# Gini is 1 = (1/16)^2 + (15/16)^2 = 0.12
# information: =1/16*log2(1/16 + 15/16*log2(15/16)) = 0.34

## Iris set
data(iris); library(ggplot2); library(caret)
table(iris$Species)
# splitting data
inTrain <- createDataPartition(y=iris$Species,
                               p=0.7, list = FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training)

# plot everything
qplot(Petal.Width, Sepal.Width, colour = Species, data = training)

# fit model using train function
modFit <- train(Species ~., method="rpart", data = training)
print(modFit$finalModel)

# plot the tree
plot(modFit$finalModel, uniform = TRUE,
     main = "Classification Tree")
text(modFit$finalModel, use.n=TRUE, all=TRUE,cex=0.8)

# fancier plot with rattle package
library(rattle)
fancyRpartPlot(modFit$finalModel)

# Predicting new values
predict(modFit, newdata = testing)

# Notes
Classification trees are non-linear models
They use interactions between variables
Data transformations may be less important
Trees can also be used for regression problems
party, rpart and tree as packages to use

## Bagging (bootstrap aggregating)
# bootstrap aggregating
Fit complicated models, if you average them it gets a better
model. 

## Basic idea
# 1) resample cases and recalculate predictions
# 2) Average or majority vote
## Notes
# similar bias
# reduced variance
# more useful for non-linear functions

## Ozone data
library(ElemStatLearn); data(ozone, package = "ElemStatLearn")
ozone <- ozone[order(ozone$ozone),]
head(ozone)
# Bagged loess
ll <- matrix(NA, nrow=10, ncol = 155)
for (i in 1:10){
  ss <- sample(1:dim(ozone)[1], replace = T)
  ozone0 <- ozone[ss,]; ozone0 <- ozone0[order(ozone),]
  loess0 <- loess(temperature ~ ozone, data = ozone0, span = 0.2)
  ll[i,] <- predict(loess0, newdata=data.frame(ozone=1:155))
}
# now plot it
plot(ozone$ozone, ozone$temperature, pch=19, cex=0.5)
for(i in 1:10){lines(1:155,ll[i,], col = "grey",lwd = 2)}
lines(1:155,apply(ll,2,mean), col = "red",lwd = 2)
# red line is bagged loess average curve

## Bagging in caret
# bagEarth, treebag, bagFDA
# or make your own bagging function
predictors = data.frame(ozone = ozone$ozone)
temperature = ozone$temperature
treebag <- bag(predictors, temperature, B = 10,
               bagControl = bagControl(fit = ctreeBag$fit,
                                       predict = ctreeBag$pred,
                                       aggregate = ctreeBag$aggregate))
plot(ozone$ozone, temperature, col = "lightgrey", pch = 19)
points(ozone$ozone, predict(treebag$fits[[1]]$fit,predictors), pch = 19, col = "red")
points(ozone$ozone, predict(treebag, predictors), pch = 19, col = "blue")
## very useful for non linear models
# often used with trees an extension is random forests
## further resources
http://en.wikipedia.org/wiki/Bootstrap_aggregating
http://stat.ethz.ch/education/semesters/FS_2008/CompStat/sk-ch8.pdf

## Random forests
# 1) bootstrap samples
# 2) at each split, bootstrap variables (only subset is considerd)
# 3) grow multiple trees and vote

## Pros
# accuracy
## Cons
# 1) speed
# 2) interpretability
# 3) Overfitting

## Example
data(iris); library(ggplot2)
library(caret)
inTrain <- createDataPartition(y=iris$Species,
                               p=0.7, list = FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
library(caret)
modFit <- train(Species~., data = training, method = "rf",prox = TRUE)
modFit

# look at specific tree in the model
getTree(modFit$finalModel, k = 2)

## Centers information what the centers of class predictions are
irisP <- classCenter(training[,c(3,4)], training$Species, modFit$finalModel$prox)
irisP <- as.data.frame(irisP); irisP$Species <- rownames(irisP)
p <- qplot(Petal.Width, Petal.Length, col = Species, data = training)
p + geom_point(aes(x = Petal.Width, y = Petal.Length, col = Species), size = 5, shape = 4, data = irisP)

# Predicting new values
pred <- predict(modFit, testing); testing$predRight <- pred==testing$Species
table(pred, testing$Species)

## Boosting
# 1) take lots weak predictors
# 2) weight them and add them up
# 3) Get a stronger predictor

## Basic idea 
# start with a set of classifiers h1, hk
# create a classifier that combines classification functions
# goal is to minimize error on trianing set
# iterative, select one h at each step
# calculate weights based on errors
# upweight missed classifications and select next h

## Example
library(ISLR); data(Wage); library(ggplot2); library(caret)
Wage <- subset(Wage, select=-c(logwage))
inTrain <- createDataPartition(y=Wage$wage,
                               p = 0.7, list = FALSE)
training <- Wage[inTrain,]; testing <- Wage[inTrain,]
modFit <- train(wage ~., method = "gbm", data = training, verbose = FALSE) 
# verbose = TRUE, a lot of output
qplot(predict(modFit, testing), wage, data = testing)

## Notes and further reading
http://www.cc.gatech.edu/~thad/6601-gradAI-fall2013/boosting.pdf
http://webee.technion.ac.il/people/rmeir/BoostingTutorial.pdf
http://www.netflixprize.com/assets/GrandPrize2009_BPC_BigChaos.pdf
https://kaggle2.blob.core.windows.net/wiki-files/327/09ccf652-8c1c-4a3d-b979-ce2369c985e4/Willem%20Mestrom%20-%20Milestone%201%20Description%20V2%202.pdf

### LEcture 5: Model based prediction
# 1) Assume the data follow a probabilistic model
# 2) Use bayes theorem to identify optimal classifiers
## Pros
# can take advantage of structure of data
# may be computiontailly convenient
# reasonablyaccurate on real problems
## Cons
# Make additional assupmtions about the data
# When model in incorrect you may get reduced accuracy

## Model based approach
# 1) goal P(Y = k| X = k)
# 2) PR(Y = k| X = x) / Pr(X = k | Y = l)
# 3) typically prior probabilities pi are set in advance
# 4) Common choice for fkx = 
# 5) Estimate the parameters, muk and sigmak from data
# 6) Classify to the class with the higghest value of P(Y = k | X = x)

## range of models use this approach
# Linear discriminant analysis fk(x) multivariate Gaussian
# Quadratic discriminant analysis (allows different covariance matrices)
# Model based prediction 
# Naive Bayes independence between the features

## Further reading
http://www.stat.washington.edu/raftery/Research/PDF/fraley2002.pdf
http://en.wikipedia.org/wiki/Linear_discriminant_analysis
http://en.wikipedia.org/wiki/Quadratic_classifier

### Quiz 3
library(AppliedPredictiveModeling)
data(segmentationOriginal)
head(segmentationOriginal)
library(caret)
training <- segmentationOriginal[segmentationOriginal$Case == "Test",]
testing <- segmentationOriginal[segmentationOriginal$Case == "Train",]
set.seed(125)
modFit <- train(Class ~ ., data = training, method = "rpart")

# New data to predict  
TotalIntench2 = 23000; FiberWidthCh1 = 10; PerimStatusCh1=2 
TotalIntench2 = 50000; FiberWidthCh1 = 10;VarIntenCh4 = 100 
TotalIntench2 = 57000; FiberWidthCh1 = 8;VarIntenCh4 = 100 
FiberWidthCh1 = 8;VarIntenCh4 = 100; PerimStatusCh1=2 
# Make the plot
library(rattle)
fancyRpartPlot(modFit$finalModel)

# Q2
smaller, smaller equal to 1

# Q3
library(pgmm)
data(olive)
olive = olive[,-1]
inTrain <- createDataPartition(y=olive$Area, list = FALSE)
training <- olive[inTrain,]
testing <- olive[-inTrain,]
newdata = as.data.frame(t(colMeans(olive)))

modFit <- train(Area ~ ., data = training, method = "rpart")
predict(modFit, newdata)

# Q4
library(ElemStatLearn)
data(SAheart)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]
set.seed(13234)

# misclassification function
missClass = function(values,prediction){
  sum(((prediction > 0.5)*1) != values)/length(values)
}

# make a model
modFit <- train(chd ~ age + alcohol + obesity + tobacco + typea + ldl, 
                data = trainSA, method = "glm", family = "binomial")
plda <- predict(modFit, testSA)
pld_train <- predict(modFit, trainSA)
table(pld_train, trainSA$chd)
missClass(trainSA$chd, as.numeric(pld_train)) #train set misclassification
table(plda, testSA$chd) # test set misclassification
missClass(testSA$chd, as.numeric(plda))


# Q5
library(ElemStatLearn)
data(vowel.train)
data(vowel.test)
vowel.train$y <- factor(vowel.train$y)
vowel.test$t <- factor(vowel.test$y)
set.seed(33833)

modFit <- train(y ~ . , data = vowel.train, method = "rf", prox = TRUE)
# calculate variable importance
varImp(modFit)

### Regularized regression
1. FIt a regression model
2. Penalize (or shrink) large coefficients
Pros
- can help with bias / vairance
- help with model selection
Cons
- computationally demanding

# Example
Y = B0 + B1X1 + B2X2 + e
library(ElemStatLearn); data(prostate)
str(prostate)

As number of varaibles increases, training set error goes down
Test set this goes up eventually

# Hard thresholding
MOdel Y = F(X) + e

## Regularization for regression
# Bjs are unconstrained
# They can explode and hencea re susceptible to very high variance
# To control variance, we might regularize / shrink the coefficients

## Ridge regression
# Fit regression model
# lambda * SUM(BJ)^2 -> Requires that some of Bjs are small
# if l = 0 -> Least squares solution
# lambda is tuning parameter

## Lasso
# shrinks some of the coefficients to zero, and some not
# so it performs model selection

### Combining predictors
# You can combine classifiers by averaging / voting
# Combining classifiers improves accuracy
# Reduce interpretability
# Boosting, bagging and random forest are all variants on this theme

## Basic idea
# 5 completely independent classifiers
# accuracy 70% -> 

## Approaches for combining classifiers
# Boosting, bagging
# Model stacking and ensembling

##  Build two different models
library(ISLR); data(Wage); library(ggplot2); library(caret)
inBuild <- createDataPartition(y = Wage$wage,
                               p = 0.7, list = FALSE)
validation <- Wage[-inBuild,]; buildData <-Wage[inBuild,]
inTrain <- createDataPartition(y = buildData$wage,
                                 p = 0.7, list = FALSE)
training <- buildData[inTrain,]; testing <- buildData[-inTrain,]
mod1 <- train(wage ~., method = "glm", data = training)
mod2 <- train(wage ~. , method = "rf",
              data = training, trControl = trainControl(method = "cv"), number = 3)
pred1 <- predict(mod1, testing); pred2 <- predict(mod2, testing)
qplot(pred1, pred2, colour = wage, data = testing)

# fit model that combines predictions
predDf <- data.frame(pred1, pred2 , wage = testing$wage)
combModFit <- train(wage ~., method = "gam", data = predDF)
combPred <- predict(combModFit, predDF)

# Testing errors
sqrt(sum(pred1 - testing$wage)^2)
sqrt(sum((pred2 - testing$wage)^2))
sqrt(sum((combPred - testing$wage)^2))

# Predict on validation data
pred1V <- predict(mod1, validation); pred2V <- predict(mod2, validation)
predVDF <- data.frame(pred1 = pred1V, pred2 = pred2V)
combPredV <- predict(combModFit, predVDF)

### Forecasting
#Data dependent over time
#Specifc pattern types
# Trends, seasonal patterns, cycles

# Google data
library(quantmod)
from.dat <- as.Date("01/01/08", format = "%m/%d/%y")
to.dat <- as.Date("12/31/13", format = "%m/%d/%y")
getSymbols("GOOG", src = "google", from = from.dat, to = to.dat)
mGoog <- to.monthly(GOOG)
googOpen <- Op(mGoog)
ts1 <- ts(googOpen, frequency = 12)
plot(ts1, xlab = "Years + 1", ylab = "GOOG")

plot(decompose(ts1), xlab = "Years+1")
# Test set
ts1Train <- window(ts1, start = 1, end = 5)
ts1Test <- window(ts1, start = 5, end = (7-0.01))
ts1Train

# Simple moving average
# Exponential smoothing
## More information
https://en.wikipedia.org/wiki/Forecasting
https://www.otexts.org/fpp/
https://cran.r-project.org/web/packages/quantmod/quantmod.pdf
https://www.quandl.com/help/packages/r

## Unsupervised prediction
data(iris); library(ggplot2)
inTrain <- createDataPartition(y = iris$Species,
                               p = 0.7, list = FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training); dim(testing)

kMeans1 <- kmeans(subset(training, select=-c(Species)), centers = 3)
training$clusters <- as.factor(kMeans1$cluster)
qplot(Petal.Width, Petal.Length, colour = clusters, data = training)
table(kMeans1$cluster, training$Species)
modFit <- train(clusters ~., data = subset(training, select= -c(Species)), method = "rpart")
table(predict(modFit, training), training$Species)
testClusterPred <- predict(modFit, testing)
table(testClusterPred, testing$Species)


### Quiz
# Q1
library(ElemStatLearn)
data(vowel.train)
data(vowel.test)
vowel.train$y = as.factor(vowel.train$y)
vowel.test$y = as.factor(vowel.test$y)

# fit models
set.seed(33833)
modFit1 <- train(y ~., data = vowel.train, method = "rf")
modFit2 <- train(y ~., data = vowel.train, method = "gbm")
pred1 <- predict(modFit1, vowel.test)
pred2 <- predict(modFit2, vowel.test)

# Pred right
vowel.test$predRight <- pred1==vowel.test$y
vowel.test$predRight2 <- pred2==vowel.test$y
sum(vowel.test$predRight) / nrow(vowel.test)
sum(vowel.test$predRight2) / nrow(vowel.test)
new <- subset(vowel.test, predRight == TRUE & predRight2 == TRUE)
nrow(new) / nrow(vowel.test)

table(pred1, vowel.test$y)
table(pred2, vowel.test$y)

# Q2
library(caret)
library(gbm)
set.seed(3433)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
set.seed(62433)

# models
mod1 <- train(diagnosis ~., training, method = "rf")
mod2 <- train(diagnosis ~., training, method = "gbm")
mod3 <- train(diagnosis ~., training, method = "lda")

pred1 <- predict(mod1, testing)
pred2 <- predict(mod2, testing)
pred3 <- predict(mod3, testing)
predDF <- data.frame(pred1, pred2, diagnosis = testing$diagnosis)
combModFit <- train(diagnosis ~., method = "rf", data = predDF)
combPred <- predict(combModFit, predDF)

# Q3
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

# Q4
library(lubridate)  # For year() function below
dat = read.csv("~/Desktop/gaData.csv")
training = dat[year(dat$date) < 2012,]
testing = dat[(year(dat$date)) > 2011,]
tstrain = ts(training$visitsTumblr)

# Q5
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]
