#### Introduction to data
#Research question:
Are consumers of certain alcohol brands more likely to end up in the emergency?

# population
Everyone

# Sample
Patients at John Hopkins University

# Generalize
Populatin of Baltimore

## Data basics
data matrix
each row represents a case
each column represents are variable

## Variables
numerical vs categorical
numerical further distributed in continuous and discrete
Count data is an example of discrete variables

categorical can be distributed in:
  - ordinal: inherent ordening
  - regular categorical

## relationships between variables
association (dependent)
not associated (independent)

## observational studies and experiments
# observational study
collect data in a way that does not directly intefrere with how data arise
only establish an associatin
if data from past is used: retrospective study
prospective: data are collected throughout study

# experiment
randomly assign subjects to treatments
establish causal relationship between intervention and outcome

## Sampling & sources of bias
# census
sample the entire population

# convenience sample
Individuals who are easily accessible

# non-response
if only a fraction of randomly sampled people respond to survey

# voluntary response
occurs when the sample consists of people who volunteer

## Sampling methods
# simple random sample (SRS)
randomly select casus from population such that each case is equally likely

# stratified sample
divide popoulation into homogenous strate and then randomly from each stratum

# cluster sampling
dividie population into clusters, randomly sample clusters and then
sample randomly observations within these clusters.
clusters are heterogenous

## Experimentals desing
control: compare treatment to control group
randomize: randomly assig subjects to treatments
replicate: collect a sufficiently large sample or replicate entire study
block: block for variable known or suspected to affect outcome

# Placebo
fake treatment often used as the control gropu for medical studies

# blinding
experimental units do not know what group they are in

# double-blind
experimental and participants do not know what group they are in

#### Part 2: Visualization numerical data
# scatterplots for paired data
# other visualization for describing distributions of numerical variables

## Two numerical variables
use a scatterplot
explanatory vs response variable

# evaluating the relationships between the two
direction -> increase vs decrease
shape -> linear or curvy
strength -> strong or weak
outliers

## Distribution ofn umerical variables
histogram: provide view of the data density
useful for describing shape (skewed or not)
modality: uni-modal vs bi-modal
bin-width can change interpretation

# Modality
bi-modality may indicate that there are two distinct groups

# uniform
no apparent trend in the data

# Dotplot
usueful when individual values are of interest
increased sample size -> it may get to busy

# Boxplot
good for visualizing outliers
median
IQR
good for skewness. Peak for distribution will be around the median

# Intensity map
shows spatial distribution

### Measures of center
shape: left skewed, symmetric and right skewed
measures of center: mean, median, mode
mode: most common value
median: midpoint of ordered data

## Measures of spread
range: difference between max and min
variance
standard deviation
inter-quartile range

# variance
average squared deviation from the mean
s^2 = SUM(xi - xbar)^2 / n-1
units of variance are squared units of the original data
so it may not be very meaningful

# standard deviation
average deviation around the mean
s and sigma for population sd
s = sqrt(s^2)

# interquartile range
range of the middle 50% of the data
IQR = Q3 - Q1

## Robust statistics
robust statistics as measures on which extreme observations have little effect

## Transforming data
# Define a transformation
rescaling of the data using a function
data strongly skewed sometimes transfer them to make modelling easier

# Log transformation
often applied when much of the data clustered near zero
can also be applied to both variables such that linear methods are possible

# square root

# inverse

# Goal transformations
1. see data structure differently
2. reduce skew to assis in modeling
3. straighten a nonlinear relatinoship in scatterplot

## exploring categorical variables
barplots: used for displaying distributions of categorical variables
x-axis order is interchangeable

# pie chart?
much less informative than a barplot. Stick to barplots

# Contingency table
displays relationships between two categorical variables

# segmented bar plot
useful for visualizaing conditional frequency distributions
compare relative frequencies: Plot them in a barplot

# mosaiplot
shows marginal distribution by width of the bars
height of the bar shows percentage of y axis

# side-by-side box plots
shows two distribution conditional on a categorical variable.names(
  
## Introduction to inference
48 male banks superviosr given personall file for promotion
files were identical except for gender
random assignment
35 / 48 promoted
  
# conditiona on gender
% males 21 / 24
%females 14 / 24

# two expalantions
H0: theres notghin going on. They are independent
Ha: Theres gender discrimination
evaluate probability of observing an outcome at least as extreme as the one observed in theo rigina data
if that probability is low, reject the null hypothesis


#### Week 2: Probablikty
P(A) = Probability of event A
0 <= P <= 1

# Bayesian probability
probability as a subjective degree of belief

# frequentist interpretation
probability of an outcome is the proportion of times the outocme woudl occur if 
we observed the random process an infinite number of times

# law of large numbers
as more observations are collected proportion of occurences with a particular outcome
will converge to the probability of that outcome

# gambler's fallacy
Probability of head and tail will always remain 50%

# disjoint events
cannot happen at the same time (mutually exclusive)
P(A and B) = 0
P(A) or P(B) = P(A) + P(B)


# complementary events
two mutually exclusive events that add up to 1

# non disjoint events
these events can overlap. You can see that in a Ven diagram
P(A) or P(B) = P(A) + P(B) - P(A and B)

# sample space
collection of all possible outcomes of a trial
for males and females
s = {MM, MF, FM, FF}

# Probability distributions
lists all possible outcomes in the sample space and the probabilities
with which they occur

# independent events
two processes independent if knowing the outcome of one does not provide
any useful information about the outcome of the other

# product rule
if A and B are independent then P(A and B) = P(A) * P(B)

# marginal probability
what is the probability that a students objective social class is upper middleclass
50/ 98. Counts came from margins of contingency table

# joint probability
considering students that are at the intersect of objective position and subjective identity

# conditional probability
What is the probability that a student who is objecitvely in the working class
associates with middle class

# Bayes theorem
P(A | B) = P(A and B) / P(B)
Generically if P(A | B) = P(A) then A and B are independent

# bayesian inference
prior probabilities for two competing claism
0.5 that good die is on the right hand, 0.5 the bad die is on the right hand

If you rolled at number greater than 4 on the right hand, probability changes
Now becomes P(H1: good die on the right | you rolled number greater than 4)
P(Good Right & >4 right) / P(>4 Right) = (0.5 * 0.75) / ((0.5 * 0.75) +(0.5 * 0.5))
This is the posterior probability = P(hypothesis | data)

In Bayesian approach, we evaluate claims iteratively as we collect more data.
we update our prior with posterior from previous iteration

A good prior helps, prior matters less the more data you have

# examples bayesian inference
BReast cancer. Prior probabiliyt 0.017
Two completing claims:
  1. Patient has cancer
  2. Patient doesnt have cancer

Testing positive given cancer = 0.78
Testing positive given no cancer = 0.10
Our posterior now becomes 0.12. (0.017 * 0.78) / (0.017 * 0.78) + (0.983 * 0.10)

Second mamogram also yields positive.
(0.12 * 0.78) / ((0.12 * 0.78) + (0.88 * 0.10))
Posterior now is 0.52

# Normal distribution
unimodal and symmetric (Bell curve)
68, 95, 99.7 (within 3sd)

# Standardizing with Z scores
Z score of an observations is number of sd away from the mean

# percentile
percetnage of observation that fall below a given data point
area below probability distribution curve left to that observation
pnorm does this in R

# binomial distribution
Milgram experiment
P (shock) = 0.65.

describes the probability of having k successes in n indepdent Bernouilli trials
with probability of success p
p^k(1- p )^(n - k)

(4 choose 1) = 4! / 1! * (4 - 1)! = 4

# example
k = 2, n = 9
How many scenarios?
choose(9, 2) = 36

# example
n = 10
p = 0.13
1 - p = 0.87
k = 8

# Normal approximation of binomial distirubtion
Recent facebook study
25% power users
average facebook user 245 friends
P(70 or more power user friends?)
p = 0.25; n = 245
mean = 0.25 * 245; Sd = sqrt(n * p * (1 - p))
sd = sqrt(245 * 0.25 * 0.75) = 6.78

### Week 3: Foundations for inference

## Sampling variability & CLT
population -> Take random sample -> Calculate sample statistic

# centra limit theories
distribution of sample statistics is nearly normal
centered at the population mean and with a standard deviation equal to
the standard deviation / sqrt(sample size)

As the sample size increases, samples should consist more consistent sample means
and hence standard error of these samples should be lower

# Conditions for CLT
1. Independence
- sampled observations must be independent
2 Sample size / skew
- either the population is normal or the population distirubtion is skewed,
then the smaple size is larger (n > 30)

## Examples
# E1
3000 songs Ipod, mean length 3.45 min, sd = 1.63
Probability randomly selected song > 5 min
Eyballing heights of histogram

# E2
I drive 6 hours, Make random playlist 100 songs. 
Probability that song lists is 360 min?
Average should be 3.6 min or greater
Se = sigma / sqrt(n) = 1.63 / sqrt(n) = 0.163, mu = 3.45
Z score = 3.6 - 3.45 / 0.163 = 0.92

# Confidence interval for a mean
CLT x ~ N(mean = mu, SE = sigma / sqrt(n))
95% confidence interval = xbar +/- 2*SE
COnfidence interval is about population mean, not sample mean

# conditions for this confidence interval
1. Independence
2. Sample size / skew: n> 30 larger if the population is highly skewed

# finding critical value of 95% confidence interval
qnorm

## accuracy vs precision
# confidence level gets bigger -> accuracy increases
-> precision decreases

Suppose we took many samples and built a confidence interval from each sample:
point estimate +/- 1.96 * SE
About 95% of those intervals would contain true population mean

Increase sample size to:
  1. Gain higher precision
  2. Gain higher accuracy

# Margin of error
ME = z * sigma / sqrt(n)
25 = 1.96 * 300 / sqrt(n) -> 25 * sqrt(n) = 1.96 * 300 = sqrt(n) = (1.96 * 300) / 25 -> n = ((1.96 * 300) / 25)^2
Make interval 1/3 of original interval -> Increase sample size 9 *n

# Exam
mu = 10
m = 9.51
sd = 4.65
n = 40
se = sd / sqrt(n)
z = (m - mu) / se

# Spam
spam = 0.09
nospam = 0.91
spam_given_spam = spam * 0.9
spam_given_nospam = nospam * 0.02
prob_spam = spam_given_spam + spam_given_nospam

# Given flag spam, prob is spam?
spam_given_spam / prob_spam

# median blood
change = c(-5, -4, -3, -2, 1, 7, 10, 11, 17, 18)
median(change)

# calculate
conf = 0.8
sd = 18
m_error should be 4
qnorm = 1.28
ME = z * sigma / sqrt(n)
4 = 1.28 * 18 / sqrt(n)
4 * sqrt(n) = 1.28 * 18
sqrt(n) = (1.28 * 18) / 4
n = ((1.28 * 18) / 4) ^2
