# Enable parallel computing
install.packages('doParallel')
library(doParallel) 
registerDoParallel()

# Create path and load data in a data frame
data.dir <- 'C:\\Users\\lucas\\Documents\\Job\\DataScience\\facial_recognition\\facial-keypoints-detection\\'
train.file <- paste0(data.dir, 'training.csv')
test.file  <- paste0(data.dir, 'test.csv')

# Transform data
d  <- read.csv(train.file, stringsAsFactors=F)
im <- foreach(im = d$Image, .combine=rbind) %dopar% {
    as.integer(unlist(strsplit(im, " ")))
}
d_submission  <- read.csv(test.file, stringsAsFactors=F)
im_submission <- foreach(im = d_submission$Image, .combine=rbind) %dopar% {
    as.integer(unlist(strsplit(im, " ")))
}
d$Image <- NULL
d_submission$Image <- NULL
set.seed(0)
idxs     <- sample(nrow(d), nrow(d)*0.8)
d.train  <- d[idxs, ]
d.test   <- d[-idxs, ]
im.train <- im[idxs,]
im.test  <- im[-idxs,]
rm("d", "im")

# Save data
save(d.train, im.train, d.test, im.test, file='data.RData')
save(d_submission, im_submission, file='data_submission.RData')
