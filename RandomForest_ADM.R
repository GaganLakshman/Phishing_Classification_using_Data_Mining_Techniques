library(caret)
library(caTools)
library(randomForest)
library(dplyr)
library(ggplot2)


#setting the seed
seed <- 1337

#Reading the data into factors as all variables are factors
data <- read.csv("dataset.csv",colClasses = "factor")

#Removing the Unique value column : Index Id 
data <- data[,-1]

#Checking for null values
which(is.na(data))

#Splitting the dataset into training and testing Seed : 1337
split = sample.split(data$Result, SplitRatio = 0.70)
train = subset(data, split == TRUE)
test = subset(data, split == FALSE)

#Benchmark model : Model considering all the websites as phishing
PhishingModel <- rep("-1",length(test$Result))
PhishingModel <- factor(PhishingModel, levels = c("-1", "1"), labels = c("-1", "1"))
table(test$Result,PhishingModel)
confusionMatrix(PhishingModel,test[,31])                                      #44.3%

#RF model with default parameters
classifier = randomForest(x = train[,-31],
                          y = train$Result,
                          ntree = 500)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test[,-31])

# Making the Confusion Matrix
cm = table(test[,31], y_pred)
confusionMatrix(y_pred, test[,31])                                           #96.47%

#Tunning Paratmeters
# 1 - Tune using 'caret' package
# Random Search
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
set.seed(1337)
mtry <- sqrt(ncol(train[,-31]))
rf_random <- train(Result ~ ., data=train, method="rf", metric="Accuracy", tuneLength=15, trControl=control)
print(rf_random)
plot(rf_random)
y_pred_rs = predict(rf_random, newdata = test[,-31])
# Making the Confusion Matrix
cm = table(test[,31], y_pred_rs)
confusionMatrix(y_pred_rs, test[,31])         #97.14 kappa:94.18

# Grid Search
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
set.seed(seed)
tunegrid <- expand.grid(.mtry=c(1:30))
rf_gridsearch <- train(Result ~ ., data=train, method="rf", metric="Accuracy", tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)
y_pred_gs = predict(rf_random, newdata = test[,-31])
# Making the Confusion Matrix
cm = table(test[,31], y_pred_gs)
confusionMatrix(y_pred_gs, test[,31])         #97.14  94.18


# 2 - Tune using algorithm tools
# Algorithm Tune (tuneRF)
set.seed(seed)
bestmtry <- tuneRF(train[,-31], train[,31], stepFactor=1.5, improve=1e-5, ntree=500,plot = TRUE)
print(bestmtry)

#Tune the RF model with mtry =10
rfModel_new <- randomForest(Result ~., 
                            data = train, 
                            ntree = 500, 
                            mtry = 10, 
                            importance = TRUE, 
                            proximity = TRUE)
                            
print(rfModel_new)
plot(rfModel_new)


# Predicting the Test set results
y_pred1 = predict(rfModel_new, newdata = test[,-31], na.action = na.pass)


# Making the Confusion Matrix
cm = table(test[,31], y_pred1)
confusionMatrix(y_pred1, test[,31])
#97.17% kappa : 94.24

# Feature Importance
#Sorts by variable importance and relevels factors to match ordering

var_importance <- data_frame(variable=setdiff(colnames(data[,-31]), "Result"),
                             importance=as.vector(importance(classifier)))
var_importance <- arrange(var_importance, desc(importance))
var_importance$variable <- factor(var_importance$variable, levels=var_importance$variable)

p <- ggplot(var_importance, aes(x=variable, weight=importance, fill=variable))
p <- p + geom_bar() + ggtitle("Variable Importance from Random Forest Fit")
p <- p + xlab("Attribute") + ylab("Variable Importance (Mean Decrease in Gini Index)")
p <- p + scale_fill_discrete(name="Variable Name")
p + theme(axis.text.x=element_blank(),
          axis.text.y=element_text(size=12),
          axis.title=element_text(size=16),
          plot.title=element_text(size=18),
          legend.title=element_text(size=16),
          legend.text=element_text(size=12))


