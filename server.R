# Load twitter libraries
library(twitteR)
library(ggmap)
library(stringr)

# provide tokens
api_key = "88HwS9EIPF5LVfxaKNAv5z3A3"
api_secret = "2gF8su7dloNs7eeHJH6KMrOjePtODM2kU9F9Pg0U4YlJwdhkOb"
access_token = "4259761636-U2QmIHXBosoUHEttTqK8fNvXjuQXVno82OOmDhq"
access_token_secret = "JOANatMvaSqclNyO8YYd0t8D9QrkHUYXj03hRKN7j1nMx"

# set up authentication
setup_twitter_oauth(api_key, api_secret, access_token, access_token_secret)

# for simplicities' sake, only allow user to select 4 locations
avail_trends = availableTrendLocations()
avail_trends = avail_trends[avail_trends$name %in% c("Amsterdam","London","New York","Singapore"),]

shinyServer(function(input, output, session) {
  # returns location input as selected by the user
  fetchLocation <- reactive({
    input$location
  })
  
  # fetches the top 5 trends based on the input location
  fetchTrends <- reactive({
    if (fetchLocation() == ""){
      out <- NULL
    }
    else{
      out <- tryCatch(getTrends(avail_trends$woeid[avail_trends$name == fetchLocation()]),
                   error = function(e) return (NULL))
    }
    return (out)
  })
  
  # a function that returns to the user interface a dropdown menu with top 5 trends
  output$tagList <- renderUI({
    varNames = fetchTrends()$name
    print (varNames)
    varNames = varNames[1:5]
    
    if (is.null(varNames)){
      varnames = "N/A"
    }
    out <- list(
      selectInput(
        inputId = "trend",
        label = "Select one of the top 5 tags",
        choices = as.list(varNames)
      ),
      tags$br()
    )
    return (out)
  })
  
  # a reactive function which grabs the hashtag trend selected by the user
  fetchTag <- reactive({
    input$trend
  })
  
  # a function that fetches 150 tweets with hashtag
  fetchTweets <- reactive({
    tweets <- searchTwitter(fetchTag(), n=150)
    tweets.df <- twListToDF(tweets)
    return (tweets.df$text)
  })

  
  # a function that edits edits all the tweats being selected by the user
  editTweets <- reactive({
    # takes a dependency of our go button
    input$goButton
    
    # make sure it doesn't run until go button is pressed
    # this prevents dependency on the fetchTweets function which is rather slow
    isolate(
      withProgress({
        setProgress(message = "Processing tweets")
        text = fetchTweets()
        text=str_replace_all(text,"[^[:graph:]]", " ")
        if (!is.null(text)){
          myCorpus = Corpus(VectorSource(text))
          myCorpus = tm_map(myCorpus, content_transformer(tolower))
          myCorpus = tm_map(myCorpus, removePunctuation)
          myCorpus = tm_map(myCorpus, removeNumbers)
          myCorpus = tm_map(myCorpus, removeWords, stopwords("english"))
          myDTM = TermDocumentMatrix(myCorpus,
                                 control = list(minWordLength = 1))
          m = as.matrix(myDTM)
        
          m = sort(rowSums(m), decreasing = TRUE)
        } else {
          m = NULL
        }
        return(m)
      
    })
    )
    
    
    
  })
  wordcloud_rep <- repeatable(wordcloud)
  
  output$plot <- renderPlot({
    if (input$goButton == 0){
      return()
    } else {
      v <- editTweets()
      wordcloud_rep(names(v), v, scale=c(4,0.5),
                    min.freq = input$frequency, max.words=input$maxTerms,
                    colors=brewer.pal(8, "Dark2"))
    }
  })
})