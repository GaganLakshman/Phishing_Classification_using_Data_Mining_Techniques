# Data the dataset
data <- read.csv("dataset.csv",colClasses = "factor")
data <- data[,-1]

data$Result<- factor(gsub("-1", "0", data$Result))
data$Result<- factor(gsub("1", "1", data$Result))

#One hot encoding
dmy <- dummyVars("~. -1",data = data)
tsf <- data.frame(predict(dmy ,newdata = data))
#Keep Result 1 from tsf, and delete the Result 0(69th column)
#0 Phishing, 1 Legitimate
tsf <- tsf[,-69]

tsf$Result.1 <- factor(tsf$Result.1)

set.seed(1337)  # For reproducibility
# Create index for testing and training data
inTrain <- createDataPartition(y = tsf$Result.1, p = 0.7, list = FALSE)
# subset data to training
train <- tsf[inTrain,]
# subset the rest to test
test <- tsf[-inTrain,]

train_matrix = xgb.DMatrix(as.matrix(train %>% select(-Result.1)))
train_label = train$Result.1

test_matrix = xgb.DMatrix(as.matrix(test %>% select(-Result.1)))
test_label = test$Result.1

xgb_trcontrol = trainControl(
  method = "cv",
  number = 5,  
  allowParallel = TRUE,
  verboseIter = FALSE,
  returnData = FALSE
)

# Creating the grid fucntion for the set of values which we want to consider
xgbGrid <- expand.grid(nrounds = c(100,200,400),
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree = seq(0.5, 0.9, length.out = 5),
                       ## The values below are default values in the sklearn-api. 
                       eta = 0.1,# if error tries to go less then this value stop training
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1)

#Train with XGBtree
set.seed(1337) 
xgb_model = train(
  train_matrix, train_label,  
  trControl = xgb_trcontrol,
  tuneGrid = xgbGrid,
  method = "xgbTree"
)

#Predicting the output on test data
predicted = predict(xgb_model, test_matrix)

#To find the accuracy applied confusion matrix
confusionMatrix(test_label, predicted)

#97.32 kappa : 94.56
plot(xgb_model)



