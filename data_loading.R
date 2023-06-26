# Enable parallel computing
install.packages('doParallel')
library(doParallel) 
registerDoParallel()

# Create path and load data in a data frame
data.dir <- 'C:\\Users\\lucas\\Documents\\Job\\DataScience\\facial_recognition\\facial-keypoints-detection\\'
train.file <- paste0(data.dir, 'training.csv')
test.file  <- paste0(data.dir, 'test.csv')
d.train <- read.csv(train.file, stringsAsFactors=F)

# Transform data
im.train      <- d.train$Image
d.train$Image <- NULL
im.train <- foreach(im = im.train, .combine=rbind) %dopar% {
    as.integer(unlist(strsplit(im, " ")))
}

d.test  <- read.csv(test.file, stringsAsFactors=F)
im.test <- foreach(im = d.test$Image, .combine=rbind) %dopar% {
    as.integer(unlist(strsplit(im, " ")))
}
d.test$Image <- NULL

# Save data
save(d.train, im.train, d.test, im.test, file='data.Rd')
