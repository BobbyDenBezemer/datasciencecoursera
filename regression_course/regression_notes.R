### Course notes regression
## Week 1
Francis Galtons data. Parent children dataset
Least squares = SUM(Yi - mu)^2
Least squares estimate is empirical mean

Suppose that Xi are the parents hight
Consider picking the slope Beta the minimizes least squares
SUM(Yi - Xi*Beta)^2

## Regression through the origin
B0 = 0. 
By centering the data, we are setting the origin right in the middle of the data
This makes it much easier now to do regression to the origin
You do this through subtracting the mean
Mean squared error is SUM(yi - xiB)^2

# Finding the slope in regression to the origin with centered data
install.packages("usingR")
library(usingR)
lm(I(child - mean(child)) ~ I(parent - mean(parent)) - 1, data = galton)

# Emperical variance and standard deviation
empirical mean: Xbar = 1 / n *SUM(xi)
Xi~ = Xi - Xbar, then mean(Xi~) = 0
This proces is called centering

S^2 = 1 / n - 1 * SUM(xi - Xmean)^2
S = sqrt(S^2)
Data defined by Xi / s have empirical standard deviation 1. 
This is called scaling

# Normalizaiton
Zi = Xi - Xmean / s
Normalized data are centered at 0 and have units equal to standard devation 1
Normalization makes otherwise non comparable data comparable

# Empirical covariance
COV(X, Y) = 1 / n -1 * SUM(Xi - Xmean)(Yi - Ymean)
Cor(X, Y) = COV(X, Y) / Sx*Sy

## Least squares estimation regression line
# Fitting the best line
YI is ith child heigh, Xi ith parents heigh
Consider Childs height = B0 + Parents height * B1
Minimize SUM{Yi - (B0 + B1*Xi)}^2

B1 = Cor(Y, X) * SDy / SDx
B0 = Yav - B1*Xav
# Consequences
B1 has the units of Y / X, B0 has the units of Y
The line always passed through the point (Xbar, Ybar)
The slope of the regression line with X as outcome and Y as predictor is
Cor(Y,X) / SD(X) / SD(Y)
THe slope is the same one as you would get if you would center the data
and did regression through the origin.
If you normalized the data, Xi-Xav / Sd(X), Yi-Yav / SD(y), the slope is COR(Y, X)

## Example of parents child height example again
library(UsingR)
data(galton)
library(dplyr); library(ggplot2)
freqData <- as.data.frame(table(galton$child, galton$parent))
names(freqData) <- c("child", "parent", "freq")
freqData$child <- as.numeric(as.character(freqData$child))
freqData$parent <- as.numeric(as.character(freqData$parent))
# making plot
g <- ggplot(filter(freqData, freq > 0), aes(x = parent, y = child))
g <- g + scale_size(range = c(2, 20), guide = "none")
g <- g + geom_point(colour = "grey50", aes(size = freq + 20, show_guide = FALSE))
g <- g + geom_point(aes(colour = freq, size = freq))
g <- g + scale_colour_gradient(low = "lightblue", high = "white")
g

# normal regression
y <- galton$child
x <- galton$parent
beta1 <- cor(y, x) * sd(y) / sd(x)
beta0 <- mean(y) - beta1 * mean(x)
rbind(c(beta0, beta1), coef(lm(y ~ x)))

# regression through origin yielded same slope with when you center y and x
yc <- y - mean(y)
xc <- x - mean(x)
beta1 <- sum(yc * xc) / sum(xc^2)
c(beta1, coef(lm(y ~ x))[2])

# Normalizing variables results in slope being correlation
yn <- (y - mean(y)) / sd(y)
xn <- (x - mean(x)) / sd(x)
c(cor(y, x), cor(yn, xn), coef(lm(yn ~ xn))[2]) # same as coefficient

## Technical details
B1 = cor(y, x) * Sd(y) / sd(x)

## Regression to the mean

### Quiz 1
x <- c(0.18, -1.54, 0.42, 0.95)
w <- c(2, 1, 3, 1)
z <- x * w
mean(z)
wi(xi?????)2 = 0
(xi - mu)^2 = wi
xi- mu = sqrt(wi)
-mu = sqrt(wi) - xi

# Q2
x <- c(0.8, 0.47, 0.51, 0.73, 0.36, 0.58, 0.57, 0.85, 0.44, 0.42)
y <- c(1.39, 0.72, 1.55, 0.48, 1.19, -1.59, 1.23, -0.65, 1.49, 0.05)
coef(lm(y ~ x - 1))
# Q3
data(mtcars)
coef(lm(mpg ~ wt, data = mtcars))[2]

# Q4
sdY = 2
sdX = 1
corYX = 0.5
beta1 = corYX * sdY / sdX

# Q5
1.5 * 0.4

# Q6
x <- c(8.58, 10.46, 9.01, 9.64, 8.86)
xnor <- (x - mean(x)) / sd(x)

# Q7
x <- c(0.8, 0.47, 0.51, 0.73, 0.36, 0.58, 0.57, 0.85, 0.44, 0.42)
y <- c(1.39, 0.72, 1.55, 0.48, 1.19, -1.59, 1.23, -0.65, 1.49, 0.05)
coef(lm(y ~ x))[1]

# Q8
It must be identically 0.

# Q9
x <- c(0.8, 0.47, 0.51, 0.73, 0.36, 0.58, 0.57, 0.85, 0.44, 0.42)
mean(x)

# Q10