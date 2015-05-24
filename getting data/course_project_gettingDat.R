### Run_analysis.R

setwd("C:/Users/Bobby/Documents/data_spec/getting data/data")
# Read in the train data, subject IDs and activities
data1 <- read.table("X_train.txt")
subject <-read.table("subject_train.txt")
activity <- read.table("Y_train.txt")

# make factor labels
activity_labels = c("Walking","Walking_upstairs","Walking_downstairs",
                    "Sitting","Standing","Laying")
activity$V1 = factor(activity$V1, labels = activity_labels) 

# add factor and subject to dataframe
data1 <- cbind(activity, data1)
data1 = cbind(subject, data1)
colnames(data1)[1] = "subject"

# Read in test data, subject IDS and activities
data4 <- read.table("X_test.txt")
subject2 <-read.table("subject_test.txt")
activity2 <- read.table("Y_test.txt")

# change activity to factor
activity2$V1 = factor(activity2$V1, labels = activity_labels)

# combine all the data frames
data4 <- cbind(activity2, data4)
data4 = cbind(subject2, data4)
colnames(data4)[1] = "subject"

## Now lets combine the test and train data rowwise
data_total = rbind(data1, data4)
colnames(data_total)[2] = "activity"

# now let's import the column names of the data frame
column_names <- read.table("features.txt")
column_names$V2 = as.character(column_names$V2)

# include those column names on dataframe
colnames(data_total)[3:dim(data_total)[2]] = column_names[,2]

# now set everything to to lower
colnames(data_total) = sapply(colnames(data_total), function(x) tolower(x))

# we have a duplicate column names. Remove these columns
data_total1 = data_total[,colnames(unique(as.matrix(data_total), MARGIN=2))]

## Filter the data. Only maintain columns about means and stds, activities and IDs
filter1 = grep('mean|std',colnames(data_total1))
data_total2 = data_total1[,c(1,2,filter1)]

## the names still look a bit bad. Lets make them better
names1 = colnames(data_total2)[3:dim(data_total2)[2]]

# remove all sorts of characters
names1 = gsub("(\\(\\)-)","_",names1)

# do some additional editting
names1 = gsub("-","_", names1)
names1 = gsub("\\(\\)$","", names1)

# there are still some very long column names at the end
names1 = gsub("(mean\\))","_mean)", names1)

# only number 79 has some weird formatting left. Do this manually
names1[79] = sub("(_mean\\))","_mean", names1[79])

# okay the columns with bodybody have a lot of characters without any interpunction. Add underscores
names1 = gsub("(bodybody)","_body_body_", names1)

## Set the names back to the data frame
colnames(data_total2)[3:dim(data_total2)[2]] = names1

## Now let's do some dplyr. Group things by ID and activity
library(dplyr)
data_tidy = data_total2 %>%
  select(contains("mean"), subject, activity) %>%
  group_by(subject, activity) %>%
  summarise_each(funs(mean))

## write this out as a table
write.table(data_tidy, file = "course_project_GetClean_data.txt",quote = FALSE,
            row.names = FALSE)

