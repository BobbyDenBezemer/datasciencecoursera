# reading the data
data1 <- read.table("household_power_consumption.txt", sep = ";",
                   stringsAsFactors = FALSE, skip = 1)

# adding the column names
colnames(data1) = c("Date","Time","Global_active_power",
                  "Global_reactive_power","Voltage","Global_intensity",
                  "Sub_metering_1","Sub_metering_2","Sub_metering_3")

# filter out only the data from february 1st and second
data1$Date = as.Date(data1$Date, format = "%d/%m/%Y")
data1 <- data1[data1$Date %in% as.Date(c('2007-02-01', '2007-02-02')),]

# now every variable is character, need to do some conversions
# Converting global active power to numeric
data1$Global_active_power = as.numeric(data1$Global_active_power)

# Making a composite variable from Date and Time
data1$Time3 = as.POSIXlt(strptime(paste(data1$Date, data1$Time, sep=""),
                       format = "%Y-%m-%d %H:%M"))



# Making the first plot: The histogram
hist(x = data1$Global_active_power, col = "red", breaks = 12,
     main = "Global Active Power",xlab = "Global active power (kilowatts)",
     freq = TRUE, ylab = "Frequency")

# Making the second plot: Line graph
Sys.setlocale("LC_ALL","C") # otherwise I get Dutch days
plot(x = data1$Time3, y = data1$Global_active_power, type = "l",
     xlab = "", ylab = "Global active power (kilowatts)")

# Making the third plot: Line graph with colors
par(mfrow = c(1,1))

# making the plot
plot(data1$Time3, data1$Sub_metering_1, type = "l", 
     xlab = "", ylab = "Energy sub metering") +
  # Adding the additional lines
lines(data1$Time3, data1$Sub_metering_2, col = "red", type = "l") +
lines(data1$Time3, data1$Sub_metering_3, col = "blue", type = "l")
# adding the legend
par(cex = .64)
legend("topright", lty = 1, col = c("black", "blue", "red"),text.font = 1,
       legend = c("Sub_metering_1", "Sub_metering_2", "Sub_metering_3"))

# Making the fourth plot: A facetted plot
par(mfrow = c(2, 2))
par(cex = .64)
with(data1, {
  # Making the first line graph
  plot(x = Time3, y = Global_active_power, type = "l",
       xlab = "", ylab = "Global active power")
  
  # Making the second line graph
  plot(x = Time3, y = Voltage, type = "l",
       xlab = "datetime")
  
  # Making the third line graph with 3 lines and a legend
  plot(x = Time3, y = Sub_metering_1, type = "l", 
       xlab = "", ylab = "Energy sub metering") +
    lines(x = Time3, y = Sub_metering_2, col = "red", type = "l") +
    lines(x = Time3, y = Sub_metering_3, col = "blue", type = "l")
  legend("topright", lty = 1, col = c("black", "blue", "red"), bty = "n",
         legend = c("Sub_metering_1", "Sub_metering_2", "Sub_metering_3"))
  
  # Making the fourth line graph 
  plot(x = Time3, y = Global_reactive_power, xlab = "datetime", type = "l")
})


