## Getting and cleaning data

## Lecture 1
# 1.1) getting data from the internet
if (!file.exists("data")){
  dir.create("data")
}
fileUrl <- "https://data.baltimorecity.gov/api/views/dz54-2aru/rows.csv?accessType=DOWNLOAD"
download.file(fileUrl, destfile = "./cameras.csv")
list.files()
dateDownloaded <- date()
data <- read.csv("cameras.csv")

# Reading local flat files

# 1.3) Reading excel files
# xlsx files
library(xlsx)
# read specific rows / columns
colIndex = 2:3
rowIndex = 1:4
cameraData <- read.xlsx( , sheetindex = 1, header = TRUE, colIndex = colIndex,
                        rowIndex = rowIndex)

## 1.4) Reading XML
# Two components
# Markup: Labels that give text structure
# Content: actual text of the document
library(XML)
fileUrl <- "http://www.w3schools.com/xml/simple.xml"
doc <- xmlTreeParse(fileUrl, useInternal = TRUE)
# look at rootNode. Wrapper element for entire document
rootNode <- xmlRoot(doc)
# get name out of rootNode
xmlName(rootNode) # breakfast menu
names(rootNode)

# access parts of a rootNode
rootNode[[1]]

# access first element of first rootNode 
rootNode[[1]][[1]]

# programatically extract part
xmlSApply(rootNode, xmlValue)

# get specific components of data
# get items on the menu
xpathSApply(rootNode,"//name",xmlValue)
# or get prices
xpathSApply(rootNode, "//price", xmlValue)

# Another example
fileUrl <- "http://espn.go.com/nfl/team/_/name/bal/baltimore-ravens"
doc <- htmlTreeParse(fileUrl, useInternal = TRUE)
scores <- xpathSApply(doc, "//li[@class='score']", xmlValue)
teams <- xpathSApply(doc, "//li[@class='team-name']", xmlValue)

## Should get into XML tutorials a bit more deeply
# http://www.stat.berkeley.edu/~statcur/Workshop2/Presentations/XML.pdf
# http://www.omegahat.org/RSXML/shortIntro.pdf
# http://www.omegahat.org/RSXML/Tour.pdf

# extract from doc1 all links with href attribute
url2 <- "http://www.statto.com/football/stats/germany/bundesliga/2014-2015/table"
doc1 <- htmlTreeParse(url2, useInternalNodes = TRUE)
a <- xpathApply(doc1, "//a[@href]", xmlGetAttr, "href")
xpathSApply(doc1, "//table/tr[@class = 'c0']/td[@class = 'c0']", xmlName)

## Using rvest
library(rvest)
# imagine the lego movie on IMDB
lego_movie <- html("http://www.imdb.com/title/tt1490017/")

# find selectorGadget that contains data we are looking for
lego_movie %>% 
  html_node("strong span") %>%
  html_text() %>%
  as.numeric()

# to find all the nodes that contain titleCast
lego_movie %>%
  html_nodes("#titleCast .itemprop span") %>%
  html_text()

# titles of authors posting on message board are found in third table
lego_movie %>%
  html_nodes("table") %>%
  .[[3]] %>%
  html_table()

## Setting up a connection
library(XML)
library(httr)
library(rvest)
library(magrittr)

# setup connection & grab HTML the "old" way w/httr
freak_get <- GET("http://torrentfreak.com/top-10-most-pirated-movies-of-the-week-130304/")
freak_html <- htmlParse(content(freak_get, as="text"))
# new way
freak <- html_session("http://torrentfreak.com/top-10-most-pirated-movies-of-the-week-130304/")

# extracting the "old" way with xpathSApply. Get the movie names
xpathSApply(freak_html, "//*/td[3]", xmlValue)[1:10]

# get ranking
xpathSApply(freak_html, "//*/td[1]", xmlValue)[2:11]

# get ratings
xpathSApply(freak_html, "//*/td[4]", xmlValue)

# get the links
xpathSApply(freak_html, "//*/td[4]/a[contains(@href,'imdb')]", xmlAttrs, "href")

## Now the new way with rvest and xpath
freak %>% html_nodes(xpath="//*/td[3]") %>% html_text() %>% .[1:10]

# ranking
freak %>% html_nodes(xpath="//*/td[1]") %>% html_text() %>% .[2:11]

# ratings
freak %>% html_nodes(xpath="//*/td[4]") %>% html_text() %>% .[1:10]

# get all the links to imdb
freak %>% 
  html_nodes(xpath="//*/td[4]/a[contains(@href,'imdb')]") %>% 
  html_attr("href") %>% .[1:10]

## Extracting with rvest and CSS selectors
freak %>% 
  html_nodes("td:nth-child(3)") %>% 
  html_text() %>% .[1:10]

# ranking
freak %>% 
  html_nodes("td:nth-child(1)") %>% 
  html_text() %>% .[2:11]

# get ratings
freak %>% 
  html_nodes("td:nth-child(4)") %>% 
  html_text() %>% .[1:10]

# get all links to imdb
freak %>% 
  html_nodes("td:nth-child(4) a[href*='imdb']") %>% 
  html_attr("href") %>% .[1:10]

# Building the data frame
data.frame(movie=freak %>% html_nodes("td:nth-child(3)") %>% html_text() %>% .[1:10],
           rank=freak %>% html_nodes("td:nth-child(1)") %>% html_text() %>% .[2:11],
           rating=freak %>% html_nodes("td:nth-child(4)") %>% html_text() %>% .[1:10],
           imdb.url=freak %>% html_nodes("td:nth-child(4) a[href*='imdb']") %>% html_attr("href") %>% .[1:10],
           stringsAsFactors=FALSE)

url <- "http://cran.r-project.org/web/packages/available_packages_by_date.html"
url %>%
  .(~ message(Sys.time(),": downloading")) %>%
  html() %>%
  html_nodes("td:nth-child(1)") %>%
  .(~ message("number of packages: ", length(.)))  

# another webscraping thing
library(rvest)
library(tidyr)

url <- "http://www.zillow.com/homes/for_sale/Greenwood-IN/fsba,fsbo,fore,cmsn_lt/house_type/52333_rid/39.638414,-86.011362,39.550714,-86.179419_rect/12_zm/0_mmm/"
house <- html(url) %>%
  html_nodes("article")

z_id <- house %>%
  html_attr("id")

address <- house %>%
  # take html node with following id
  html_node(".property-address a") %>%
  # take attribute href
  html_attr("href") %>%
  # split it with seperator /
  strsplit("/") %>%
  # take 3rd element
  pluck(3, character(1))

area <- function(x){
  val <- as.numeric(gsub("[^0-9.]+","",x))
  as.integer(val * ifelse(grepl("ac", x), 43560, 1))
}

sqft <- house %>%
  html_node(".property-lot") %>%
  html_text() %>%
  area()

year_build <- house %>%
  html_node(".property-year") %>%
  html_text() %>%
  gsub("Built in ","", .) %>%
  as.integer()

price <- house %>%
  html_node(".price-large") %>%
  html_text()  %>%
  extract_numeric()

params <- house %>%
  html_node(".property-data") %>%
  html_text() %>%
  strsplit(", ")

beds <- params %>%
  pluck(1, character(1)) %>%
  extract_numeric()

# use heineken experience tripadvisor
library(rvest)
url <- "http://www.tripadvisor.nl/Attraction_Review-g188590-d240813-Reviews-Heineken_Experience-Amsterdam_North_Holland_Province.html"
data <- html(url)

score <- data %>%
  html_nodes(".rating_s_fill") %>%
  html_attr("alt")



## Now on tripadvisor data
url <- "http://www.tripadvisor.com/Hotel_Review-g37209-d1762915-Reviews-JW_Marriott_Indianapolis-Indianapolis_Indiana.html"

reviews <- url %>%
  html() %>%
  html_nodes("#REVIEWS .innerBubble")

id <- reviews %>%
  html_node(".quote a") %>%
  html_attr("id")

quote <- reviews %>%
  html_node(".quote span") %>%
  html_text()

rating <- reviews %>%
  html_node(".rating .rating_s_fill") %>%
  html_attr("alt") %>%
  gsub(" of 5 stars", "", .) %>%
  as.integer()

date <- reviews %>%
  html_node(".rating .ratingDate") %>%
  html_attr("title") %>%
  strptime("%b %d, %Y") %>%
  as.POSIXct()

review <- reviews %>%
  html_node(".entry .partial_entry") %>%
  html_text()

data.frame(id, quote, rating, date, review, stringsAsFactors = FALSE) %>% View()

## A team tutorial
ateam <- html("http://www.boxofficemojo.com/movies/?id=ateam.htm")
test <- ateam %>% 
  html_nodes("table") %>% 
  # extracts first table
  extract2(1) %>% 
  html_nodes("img")

# all images in first 2 tables
ateam %>% 
  html_nodes("table") %>% 
  extract(1:2) %>% html_nodes("img")

## Another example using JSON data
var = 201401
url = paste("http://stats.grok.se/json/en/",var,"/web_scraping",sep="")
url
browseURL(url)
# Fetching data
raw.data <- readLines(url, warn = "F")
raw.data

# use package rjson
install.packages("rjson")
library(rjson)
rd <- fromJSON(raw.data)

# we only want the key daily views
rd.views <- rd$daily_views
rd.views = unlist(rd.views)
df <- as.data.frame(rd.views)

# getting started with functions
getData <- function(url){
  raw.data <- readLines(url, warn = "F")
  rd <- fromJSON(raw.data)
  rd.views <- rd$daily_views
  rd.views <- unlist(rd.views)
  rd <- as.data.frame(rd.views)
  rd$date <- rownames(rd)
  rownames(rd) <- NULL
  return (rd)
}

# Get urls for certain years
getUrl <- function(y1,y2,term){
  #function to create a list of urls given a term and a start and endpoint
  urls <- NULL
  for (year in y1:y2){
    for (month in 1:9){
      urls <- c(urls,(paste("http://stats.grok.se/json/en/",year,0,month,"/",term,sep="")))
    }
    
    for (month in 10:12){
      urls <- c(urls,(paste("http://stats.grok.se/json/en/",year,month,"/",term,sep="")))
    }
  }
  return(urls)
}

# get the statistics
getStats <- function(y1,y2,terms){
  #function to download data for each term
  #returns a dataframe
  output <- NULL
  for (term in terms){
    urls <- getUrls(y1,y2,term)
    
    results <- NULL
    for (url in urls){
      print(url)
      results <- rbind(results,getData(url))
    }
    results$term <- term
    
    output <- rbind(output,results)
  }
  return(output)
}

# visualize it
visualiseStats <- function(input){
  #function to visualise data from the getStats function
  require(lubridate)
  require(ggplot2)
  input$date <- as.Date(input$date)
  ggplot(input,aes(date,rd.views,colour=term))+geom_line()
}

input <- getStats(2011,2012,c("Data_mining","Web_scraping"))
visualiseStats(input)

## Lecture 2
# Example BBC artcle
url <- "http://www.bbc.co.uk/news/world-europe-26333587"
SOURCE <- htmlTreeParse(url, useInternal = TRUE)
# get header of the story
(xpathSApply(SOURCE,"//h1[@class='story-header']", xmlValue))

# get date of story
xpathSApply(SOURCE, "//span[@class='date']", xmlValue)

# Meta field for better formatting
xpathSApply(SOURCE, "//meta[@name='OriginalPublicationDate']/@content")

# Make a scraper
bbcScraper <- function(url){
  SOURCE <- htmlTreeParse(url, useInternal = TRUE)
  title = xpathSApply(SOURCE,"//h1[@class='story-header']", xmlValue)
  date = as.character(xpathSApply(SOURCE,
                                  "//meta[@name='OriginalPublicationDate']/@content"))
  return(c(title, date))                    
}

# Test the scraper
bbcScraper("http://www.bbc.co.uk/news/world-middle-east-26333533")

# Not all pages have meta fields. Adding exceptions
bbcScraper2 <- function(url){
  title = date = NA
  SOURCE <- htmlTreeParse(url, useInternal = TRUE)
  title = xpathSApply(SOURCE,"//h1[@class='story-header']", xmlValue)
  date = as.character(xpathSApply(SOURCE,
                                 "//meta[@name='OriginalPublicationDate']/@content"))
  if (is.null(date)){
    date = (xpathSApply(SOURCE, "//span[@class = 'date']", xmlValue))
  }
  return (c(title, as.character(date)))  
}

## Make a guardian webscraper
guardianScraper <- function(url){
  PARSED <- htmlTreeParse(url, useInternal = TRUE)
  title <- xpathSApply(PARSED, "//h1[contains(@itemprop,'headline')]",xmlValue)
  author <- xpathSApply(PARSED, "//time[@itemprop='datePublished']",xmlValue)
  time  <- xpathSApply(PARSED, "//time[@itemprop='datePublished']/@datetime")
  tags <- unique(xpathSApply(PARSED, "//a[@rel='tag']",xmlValue))
  text <- xpathSApply(PARSED, "//div[@id='article-body-blocks']/p",xmlValue)
  return(list(title=title,
              author=author,
              time=time,
              tags=paste(tags,collapse="|")
              ,text=paste(text,collapse="|")))
}

test <- guardianScraper("http://www.theguardian.com/commentisfree/2014/feb/25/how-much-cost-growers-bananas-68p-per-kilo")
test['title']


# Follow up with this stuff
http://quantifyingmemory.blogspot.nl/2014/03/web-scraping-scaling-up-digital-data.html
http://quantifyingmemory.blogspot.nl/2014/03/web-scraping-working-with-apis.html
http://www.r-bloggers.com/chartingmapping-the-scottish-vote-with-r-an-rvestdplyrtidyrtopojsonggplot-tutorial/

  ## 1.5) JSON
  library(jsonlite)
jsonData <- fromJSON("https://api.github.com/users/jtleek/repos")

# writing data frames to JSON if you want to export it to an API
myjson <- toJSON(iris, pretty = TRUE)

## More information and tutorials
# http://www.json.org/
# http://www.r-bloggers.com/new-package-jsonlite-a-smarter-json-encoderdecoder/
# http://cran.r-project.org/web/packages/jsonlite/vignettes/json-mapping.pdf

## 1.6) Data table package
library(data.table)
DF = data.table(x = rnorm(9), y = rep(c("a","b","c"), each = 3), z = rnorm(9))

# See all data tables in memory
tables()

# subsetting function is modified
DF[,w:=z^2]

# add variable that is TRUE and FALSE
DF[,a:=x>0]

# plyr like operations. Take means of x+w where a is TRUE and a is FALSE
DF[,b:= mean(x+w),by=a]

# Keys in data table
DT <- data.table(x = rep(c("a","b","c"), each = 100), y = rnorm(300))
setkey(DT, x)
# subset on value a
DT['a']

## Lecture 1) QUIZ
# 1) Download data
url <- "https://d396qusza40orc.cloudfront.net/getdata%2Fdata%2Fss06hid.csv"
data = read.csv(file = url)
download.file(url = url, destfile = "./data/test.csv", method = "curl")
data <- read.csv("getdata-data-ss06hid.csv")
tot = which(data$VAL == 24)
length(tot)

# 3) read xlsx
install.packages("xlsx")
library("xlsx")
setwd("./data")
url = "https://d396qusza40orc.cloudfront.net/getdata%2Fdata%2FDATA.gov_NGAP.xlsx"
download.file(url, destfile = "test.xlsx")
dat <- read.xlsx("test.xlsx",sheetIndex = 1,
                 header = TRUE, colIndex = 7:15,
                 rowIndex = 18:23)
sum(dat$Zip*dat$Ext,na.rm=T)

# 4) Load xml
library(XML)
fileUrl <- "http://d396qusza40orc.cloudfront.net/getdata%2Fdata%2Frestaurants.xml"
doc <- xmlTreeParse(fileUrl, useInternal = TRUE)
# look at rootNode. Wrapper element for entire document
rootNode <- xmlRoot(doc)
# get all the zipcodes
codes = xpathSApply(rootNode,"//zipcode",xmlValue)
tot = which(codes == "21231") # gives indices
length(tot)

# 5) 
url = "https://d396qusza40orc.cloudfront.net/getdata%2Fdata%2Fss06pid.csv"
DT = fread(url)
system.time(DT[,mean(pwgtp15),by=SEX]) # fastest way
system.time(tapply(DT$pwgtp15,DT$SEX,mean))
system.time(mean(DT$pwgtp15,by=DT$SEX))
system.time(sapply(split(DT$pwgtp15,DT$SEX),mean))
### LEcture 2) 

## 2.1) MySQL
# Data structured in databases, 
# each row is called record

# Install mysql on your laptop
# Use link in instructor notes

install.packages("RMySQL")

# Once loaded RmySQL use following commands:
ucscDb <- dbConnect(MySQL(), user="genome",
                    host = "genome-mysql.cse.ucsc.edu")
# run a get query
result <- dbGetQuery(ucscDb,"show databases;")
dbDisconnect(ucscDb)

# result gives all sorts of tables
hg19 <- dbConnect(MySQL(), user = "genome", db = "hg19",
                  host = "genome-mysql.cse.ucsc.edu")
# give all tables in specific database
allTables <- dbListTables(hg19)

# you want all columns in the table
dbListFields(hg19, "affyU133Plus2")

# how many rows does table have
dbGetQuery(hg19, "select count(*) from affyU133Plus2")

# get datatable
affyData <- dbReadTable(hg19, "affyU133Plus2")

# select only subset
query <- dbSendQuery(hg19, "select * from affyU133Plus2 where misMatches between 1 and 3")
affyMis <- fetch(query)
quantile(affyMis$misMatches)

# if you only want some records
affyMisSmall <- fetch(query, n = 10)
dblClearResult(query)

dbDiconnect(hg19)

# Read following posts
http://biostat.mc.vanderbilt.edu/wiki/Main/RMySQL
http://www.ahschulz.de/2013/07/23/installing-rmysql-under-windows/
http://cran.r-project.org/web/packages/RMySQL/RMySQL.pdf
http://www.pantz.org/software/mysql/mysqlcommands.html
http://www.r-bloggers.com/mysql-and-r/
  
  
## 2.2) Hdf5
# HDF5 used for storing large data sets
# install rhdf5 package
source("http://bioconductor.org/biocLite.R")
biocLite("rhdf5")

# small tutorial on rhdf5
created = h5creatGroup("example.h5","foo")
# subgroup of foo
created = h5createGroup("example.h5","foo/baa")
# to write information there
h5write(A, "example.h5", "foo/A")

# http://www.bioconductor.org/packages/release/bioc/vignettes/rhdf5/inst/doc/rhdf5.pdf

## 2.3) Reading data from the web
# Webscraping: extracting data from websites
# Example: google scholar.

con = url("http://scholar.google.com/citations?user=HI-I6C0AAAAJ&hl=en")
htmlCode = readLines(con)
close(con)

# use xpath
xpathSApply(html, "//td[@id='col-citedby']",xmlValue)

# GET from httr package
library(httr)
html2 = GET(url)

content2 = conten(html2, as="text")
parsedHtml = htmlParse(content2, asText = TRUE)
# examples
http://www.r-bloggers.com/?s=Web+Scraping
http://cran.r-project.org/web/packages/httr/httr.pdf

## 2.4) APIs
# application programming interfaces
# https://dev.twitter.com/apps
library(httr)
myapp = oauth_app("twitter",
                  key = "",secret = "")

sig = sign_oauth1.0(myapp,
                    token = "",
                    token_secret = "")
homeTL = GET("https://api.twitter.com/1.1/statuses/home_timeline.json", sig)

json1 = content(homeTL)
json2 = jsonlite::fromJSON(toJSON(json1))
# now each row corresponds to a tweet
# Use following resources
https://dev.twitter.com/rest/reference/get/search/tweets
https://dev.twitter.com/rest/public

## 2.5) Reading from other data sources
# TUtorials for other databases
http://cran.r-project.org/web/packages/RPostgreSQL/RPostgreSQL.pdf
http://cran.r-project.org/web/packages/RODBC/RODBC.pdf
http://www.r-bloggers.com/r-and-mongodb/
  
## QUIZ
  
# 2.1) 
library("httr")
library('jsonlite')
oauth_endpoints("github")
myapp<- oauth_app("github",
                  key = "31239c7b17c4ffa0587f",
                  secret = "7855110e7c344b3b7d4eb283bd72ae8c3ee69989")

github_token = oauth2.0_token(oauth_endpoints("github"),myapp)

# 4. Use API
gtoken <- config(token = github_token)
req <- GET("https://api.github.com/users/jtleek/repos", gtoken)
stop_for_status(req)

json1 = content(req)
json2 = jsonlite::fromJSON(toJSON(json1))

data_analysis_class = json2[json2$name == "datasharing",]
data_analysis_class$created

# 2.2)
library(sqldf)
url = "https://d396qusza40orc.cloudfront.net/getdata%2Fdata%2Fss06pid.csv"
download.file(url = url, destfile = "getdata1.csv")
acs <- read.csv("getdata1.csv")
text = sqldf("select pwgtp1 from acs where AGEP < 50")
head(text)

# 2.3) 
unique(acs$AGEP)
sqldf("select distinct pwgtp1 from acs")
sqldf("select distinct AGEP from acs")
text1 = sqldf("select distinct pwgtp1 from acs")
text1

# 2.4) 
con <- url("http://biostat.jhsph.edu/~jleek/contact.html")
htmlCode = readLines(con)
nchar(htmlCode[10])
nchar(htmlCode[20])
nchar(htmlCode[30])
nchar(htmlCode[100])

# 2.5) 
x <- read.fwf(
  file=url("https://d396qusza40orc.cloudfront.net/getdata%2Fwksst8110.for"),
  skip=4,
  widths=c(12, 7,4, 9,4, 9,4, 9,4))

head(x)
sum(x$V4)
sum(x$V3)
sum(x$V5)

### Week3) 

## 3.1) Subsetting and sorting
x <- data.frame("var1" = sample(1:5), "var2" = sample(6:10), "var3" = sample(11:15))
x <- x[sample(1:5),] ; x$var2[c(1,3)] = NA

# subsetting
x[,1] # get first column
x[,"var1"] # same as previous
# subset with both columns and rows
x[1:2,"var2"]

# subset logical
x[(x$var1 <= 3 & x$var3 > 11),]

# use which to return indices
x[which(x$var2 > 6),]

# sort 
sort(x$var1)
sort(x$var1, decreasing = TRUE)
sort(x$var2, na.last = TRUE)

# ordering dataframe by var1
x[order(x$var1),]

# or use plyr
library(plyr)
arrange(x, var1)
arrange(x, desc(var1)) # descending

# adding rows and columns
x$var4 <- rnorm(5)
# or use cbind
y <- cbind(x, rnorm(5))

# Summarizing data
head(restData, n = 3)
tail(restData, n = 3)
summary(restData)
str(restData)
quantile(restData$councilDistrict, na.rm = TRUE)
quantile(restData$councilDistrict, probs = c(0.5,0.7,0.9))
table(restData$zipCode, useNA = "ifany") # you get cell with number of missing values

# two dimensional table
table(restData$councilDistrict, restData$zipCode)

# check for missing value
sum(is.na(restData$councilDistrict))
any(is.na(restData$councilDistrict)) # TRUE if any value is NA
all(restData$zipCode > 0) # TRUE is all zipcodes > 0

# Row and column sums
colSums(is.na(restData)) # get columns sums of NA per column

# find all zipcodes equal to 21212 or 21213
table(restData$zipCode %in% c("21212", "21213"))

# subset by the last zipcodes
restData[restData$zipCode %in% c("21212","21213"),]

# Cross tabs
data(UCBAdmissions)
DF = as.data.frame(UCBAdmissions)
summary(DF)
# frequencies for Gender and admitted
xt <- xtabs(Freq ~ Gender + Admit, data = DF)
xt <- xtabs(breaks ~. data = warpbreaks)

# make flat table from crosstab
ftable(xt)

# size of dataset
fakeData = rnorm(1e5)
object.size(fakeData)
print(object.size(fakeData), units = "Mb")

## 3.3) Creating new variables

s1 <- seq(1, 10, by = 2) # from 1-10 by steps of 2
s2 <- seq(1, 10, length = 3) # here only 3 values
x <- c(1,3,8,25,100); seq(along = x) # vector indexes of x

# Creating binary variables
restData$zipWrong = ifelse(restData$zipCode < 0, TRUE, FALSE)

# Create categorical variable
restData$zipGroups = cut(restData$zipCOde, breaks = quantile(restDat$zipCode))

# Easier cutting
library(Hmisc)
retsData$ziproups = cut2(restData$zipCode, g = 4)

# Creating factor variables
restData$zcf <- factor(restData$zipCode)
restData$zcf[1:10]

yesno <- sample(c("yes", "no"), size = 10, replace = TRUE)
yesnofac = factor(yesno, levels = c("yes","no"))
relevel(yesnofac, ref = "yes")
# get factor back to numeric
as.numeric(yesnofac)

# Mutate function
library(dplyr)
restData2 = mutate(restData, zipGroups = 5)

# Common transformations
abs(x) # absolute value
sqrt(x) # 
ceiling(x) # ceiling(3,475) = 4
floor(x) # floor(3,8) = 3
round(x, digits = n) # round on some number of digits

## 3.4) Reshaping data
library(reshape2)
head(mtcars)
mtcars$carname = rownames(mtcars)
carMelt <- melt(mtcars, id = c("carname", "gear", "cyl"), measure.vars = c("mpg","hp"))
# id of variables, variable column is mpg and ht, and you get value column

# dcast dataset. Here dataset is summarized. 
# for 4 cylinders, we have 11 measures of mpg and horse power
cylData <- dcast(carMelt, cyl ~ variable)

# different way to summarize the data
cylData <- dcast(carMelt, cyl ~ variable, mean)

# now insectSprays dataset
# get counts for every level of factor Spray
tapply(InsectSprays$count, InsectSprays$spray, sum)

# another way is split, apply, combine
spIns = split(InsectSprays$count, InsectSprays$spray)

# apply function to list
sprCount = lapply(spIns, sum) # to get list
unlist(sprCount)
sapply(spIns, sum) # to get vector

# another way plyr
ddply(InsectSprays, .(spray), summarize, sum = ave(count, fun = sum))
# you get sum on all A values

## 3.5) Merging data
# match them based on an ID
fileUrl1 = "https://dl.dropboxusercontent.com/u/7710864/data/reviews-apr29.csv"
fileUrl2 = "https://dl.dropboxusercontent.com/u/7710864/data/solutions-apr29.csv"
download.file(fileUrl1, "./data/reviews.csv")
download.file(fileUrl2, "./data/solutions.csv")
reviews = read.csv("./data/reviews.csv"); solutions <- read.csv("./data/solutions.csv")
head(reviews, 2)
names(reviews)
names(solutions)

# merge datasets together.
# x, y , by, by.x, by.y, all
mergedData = merge(reviews, solutions, by.x="solution_id", by.y = "id", all = TRUE)

# merge all common names
mergedData2 = merge(reviews, solutions, all = TRUE)

# use join in plyr package
library(plyr)
df1 = data.frame(id = sample(1:10), x = rnorm(10))
df2 = data.frame(id = sample(1:10), y = rnorm(10))
arrange(join(df1, df2), id)

# if you have multiple data frames
df3 = data.frame(id = sample(1:10), z = rnorm(10))
dfList = list(df1, df2, df3)
join_all(dfList)

## Quiz 3
setwd("C:/Users/Bobby/Documents/data_spec/getting data")
fileUrl2 = "https://d396qusza40orc.cloudfront.net/getdata%2Fdata%2Fss06hid.csv"
download.file(fileUrl2, destfile = "Quiz3.1.csv")
test = read.csv("Quiz3.1.csv")

agricultureLogical = (test$ACR == 3 & test$AGS == 6) 
which(agricultureLogical)

## Question 2
fileUrl3 = "https://d396qusza40orc.cloudfront.net/getdata%2Fjeff.jpg"
download.file(fileUrl3,"picture.jpg")
library(jpeg)

test2 = readJPEG("picture.jpg", native = TRUE)
quantile(test2, probs = c(0.3, 0.8)) 

## QUestion 3
fileUrl4 = "https://d396qusza40orc.cloudfront.net/getdata%2Fdata%2FGDP.csv"
fileUrl5 = "https://d396qusza40orc.cloudfront.net/getdata%2Fdata%2FEDSTATS_Country.csv"

download.file(fileUrl4, "gdp.csv")
download.file(fileUrl5, "educ.csv")

gdp = read.csv("gdp.csv", skip = 4)
colnames(gdp)[1:2] = c("CountryCode", "Ranking")
selected = 1:190
gdp = gdp[gdp$Ranking %in% selected,]

educ = read.csv("educ.csv")

library(dplyr)
test2 = merge(gdp, educ, by.x = "CountryCode", all.x = TRUE)
test2$Ranking = as.numeric(as.character(test2$Ranking))
test2 = arrange(test2, desc(Ranking))
test2[190,]
nrow(test2)
merged = merge(gdp, educ, by = )

## Get average income for OECD countries
summarise(group_by(test2, Income.Group), mean(Ranking))

## cut GDP into 5 groups
library(Hmisc)
test2$groups = cut2(test2$Ranking, g = 5)
table(test2$Income.Group, test2$groups)

## Lecture 4)
# setting variables upper or lower
tolower(names(cameraData))
toupper(names(cameraData))


# splitting variable names. \\ is escape character
splitNames = strsplit(names(camerdata),"\\.")

# remove first occurence of character
sub("_","", testName)

# remove multiple occurences of character
gsub("_","".testName) # if a name has 2 underscores, both are now removed

# Finding values
grep("Alameda", cameraData$intersection) # finds indices
grep("Alameda", cameraData$intersection, value = TRUE) # get values

# check if particular vlaue appears in vector.
length(grep("JEffStreet", cameraData$intersection)) # zero. So no occurences

#  Returns boolean. So true if it appears, else FALSE
grepl("Alameda", cameraData$intersection)

# nchar. Returns number of characters in string
nchar("Bobby den Bezemer")

# take out part of the string
substr("Bobby den Bezemer",1, 5)

# paste without space
paste0("Jeffrey","Leek")

# trim your string without excess space
library(stringr)
str_trim("Bobby         ")

## 4.2) Regular expressions
^i think # ill match the start of the line
morning$ # matches the end of the line. SO morning should be at the end to match

# sets of characters
[Bb][Uu][Ss][Hh]

# Beginning of line for lower or uppercase i
^[Ii] am

# specify ranges
^[0-9][a-zA-Z]

# carrot here means NOT.  
[^?.]$ # any line that ends with anything else than a . or ? 

## 4.3) Regular expressions
. # any character
9.11 # 9, any character and then 11

# | character. Combine 2 expressions
flood|fire

# combine. 
^[Gg]ood|[Bb]ad # good at the beginning of line. Bad somewhere else

# beginning of line neds to be either good or bad
^([Gg]ood|[Bb]ad)

# ? ending line means optional. \ is escape here
[Gg]eorge( [Ww]\.)? [Bb]ush

# (.*) any character repeated any number of times.

# + means at least one of the items
[0-9]+(.*)[0-9]+
pattern = "[0-9]+(.*)[0-9]+"
x = "working as MP here 720 MP battallion, 42nd birgada"
grep(pattern, x, value = TRUE)

# {} interval. So 1-5 times
[Bb]ush( +[^ ]+ +){1,5}

# m,n means at least m but not more than n matches
# m means exactly m matches
# m, means at least m matches

# matches bla bla, It looks for repetition
pattern = +([a-zA-Z]+) +\1 +
  
# * is greedy. Always looks for longest match
^s(.*)s

# make it less greedy
^s(.*?)s$
  
## Quiz 4
# Q1
setwd("C:/Users/Bobby/Documents/data_spec/getting data")
url = "https://d396qusza40orc.cloudfront.net/getdata%2Fdata%2Fss06hid.csv"
download.file(url = url, destfile = "./data/quiz41.csv")
data = read.csv("./data/quiz41.csv")
names(data)
test = strsplit(names(data), split = "wgtp")
test[[123]]

# Q2
url = "https://d396qusza40orc.cloudfront.net/getdata%2Fdata%2FGDP.csv"
download.file(url = url, destfile = "./data/quiz42.csv")
data = read.csv("./data/quiz42.csv", skip = 4)
data = data[1:190,]
data$X.4 = gsub(pattern = ",", "", x = data$X.4)
data$X.4
data$X.4 = as.numeric(data$X.4)
mean(data$X.4, na.rm = TRUE)

# Q3
grep("^United", data$X.3)

# Q4 
url = "https://d396qusza40orc.cloudfront.net/getdata%2Fdata%2FEDSTATS_Country.csv"
download.file(url = url, destfile = "./data/quiz43.csv")
data2 = read.csv("./data/quiz43.csv")
names(data2)
data2$CountryCode = as.character(data2$CountryCode)
# merge data frame
merged = merge(data, data2, by.x = "X", by.y = "CountryCode")
head(merged)
# which hae fical end year in jun
length(grep(pattern = "[Jj]une", merged$Special.Notes))

# Q5
install.packages("quantmod")
library(quantmod)
library(lubridate)
amzn = getSymbols("AMZN",auto.assign=FALSE)
sampleTimes = index(amzn)
sampleTimes = as.Date(sampleTimes, format = "%Y-%m-%d")
sampleTimes1 = year(sampleTimes)
indice = which(sampleTimes1 == 2012)
length(indice)
sampleTimes2 = wday(sampleTimes)
sampleTimes2 = sampleTimes2[indice]
sum(sampleTimes2 == 2)
sampleTimes2
