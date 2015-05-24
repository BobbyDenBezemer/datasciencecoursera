This README file explains how the analyses were conducted. For a step to step overview of what each line of R code does, 
please read the comments in the R script.

During the analyses, the training data was imported firstly. The training data consisted of the subjects, the activities
conducted by the subjects and the data itself on the various body statistics. The activity data, subject data and bodily data 
were combined straight away. These same steps were conducted for the test data. At last the train data and test data
were combined row wise.

Subsequently, the column headers were imported. These were then applied to all columns of the the combined data frame except
the subject and activity column. Next duplicate columns were removed. 

The last part of the data wrangling analyses were the editing of the column names. ALl column names were put lowercase and 
many sort of characters were removed using regular expressions. 

Finally, the data was grouped using dplyr by subject and activity. All columns were then summarized using the mean of every
subject per activity.
