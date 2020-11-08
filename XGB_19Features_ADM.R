library(caret)
library(magrittr)
library(dplyr)
library(xgboost)
#-------------------------------- Extreme Gradient Boosting (XGBoost) ---------------------------------------------
# Data
data <- read.csv("19Features.csv",colClasses = "factor")
data <- data[,-1]

data$Result<- factor(gsub("-1", "0", data$Result))
data$Result<- factor(gsub("1", "1", data$Result))

dmy <- dummyVars("~. -1",data = data)
tsf <- data.frame(predict(dmy ,newdata = data))
#Keep Result 1 from tsf, and delete the Result 0(69th column)
#0 Phishing, 1 Legitimate
tsf <- tsf[,-46]

#tsf$Result1 <- factor(tsf$Result1)

set.seed(1337)  # For reproducibility
# Create index for testing and training data
inTrain <- createDataPartition(y = tsf$Result.1, p = 0.7, list = FALSE)
# subset data to training
train <- tsf[inTrain,]
# subset the rest to test
test <- tsf[-inTrain,]

train_matrix =as.matrix(train %>% select(-Result.1))
train_label = as.numeric(train$Result.1)

test_matrix = as.matrix(test %>% select(-Result.1))
test_label = as.numeric(test$Result.1)

xgb <- xgboost(data = train_matrix,label = train_label,
               verbose=0,
               early_stopping_rounds = 3,
               eta =0.1, # between 0 and 1
               nthread =2,
               max.depth =20,
               colsample_bytree =0.8,
               nrounds =400)

training_predict <- predict(xgb,newdata = train_matrix)
training_predict[training_predict < 0.5] <- 0
training_predict[training_predict > 0.5] <- 1

train_accuracy <- 100 * length(which(training_predict== train_label))/length(train_label)
train_accuracy

testing_predict <- predict(xgb,newdata = test_matrix)
testing_predict[testing_predict < 0.5] <- 0
testing_predict[testing_predict > 0.5] <- 1


confusionMatrix(as.factor(test_label), as.factor(testing_predict))

#----------------------------------------------------------------

#Error plot graph
e <- data.frame(xgb$evaluation_log)
plot(e$iter, e$train_mlogloss, col = 'blue')
lines(e$iter, e$test_mlogloss, col = 'red')

# Feature importance
imp <- xgb.importance(colnames(train_matrix), model = xgb)
print(imp)
xgb.plot.importance(imp)

