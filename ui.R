#setwd("C:/Users/Bobby/Documents/MOOCS/datasciencecoursera/data_products/wordcloud_app")

library(shiny)

# Define UI for application that draws a histogram
shinyUI(fluidPage(
  
  # Application title
  titlePanel("Twitter wordcloud"),
  
  # Sidebar with a slider input for the number of bins
  sidebarLayout(
    sidebarPanel(
      selectInput("location", label = h3("Select location"), 
                  choices = list("Amsterdam", "London", 
                                 "New York", "Singapore"), 
                  selected = "New York"),
      uiOutput(outputId = "tagList"),
      actionButton("goButton", "Go!"),
      sliderInput("frequency",
                  "Minimum frequency:",
                  min = 1,
                  max = 50,
                  value = 5),
      sliderInput('maxTerms',
                  'Maximum terms',
                  min = 1,
                  max = 100,
                  value = 50)
    ),
    
    # Show a plot of the generated distribution
    mainPanel(
      plotOutput("plot")
    )
  )
))