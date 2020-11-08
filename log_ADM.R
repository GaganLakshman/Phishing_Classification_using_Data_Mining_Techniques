library(caTools)
library(caret)
library(ROCR) 
library(Metrics)
library(ModelMetrics)
library(Deducer)

#Read the data
data <- read.csv("dataset.csv",colClasses = "factor")

#Remove the column with unique columns
data <- data[,-1]

#Converting -1 to 0 (Phishing) and 1 (legitimate)
data$Result<- factor(gsub("-1", "0", data$Result))
data$Result<- factor(gsub("1", "1", data$Result))

#Setting the seed so that we can reproduce the result
set.seed(1337)
split <- sample.split(data$Result,SplitRatio = .70)
train <- subset(data,split == TRUE)
test <- subset(data,split==FALSE)

#Logistic model considering all variables
model <- glm(Result ~.,data = train,family = binomial(link = 'logit'))

summary(model)

#Checking with annova to find the over all effect of the independent variables on the depended variable
anova(model,test = "Chisq")

#Checking the accuracy with full model on test data 
test_predict <- predict(model,newdata = test[,-31])
test_predict[test_predict < 0.5] <- 0
test_predict[test_predict > 0.5] <- 1

confusionMatrix(as.factor(test_predict),test[,31])  #Acc: 93.7 and Kappa = 87.26%

#StepAIC function with backward elimination, which gives the output with least AIC Model
aic_select <- stepAIC(model,direction = "backward",trace = TRUE)

#model with features with the signifcant values as the output from stepAIC , where AIC value : 2207.61 
myform <- as.formula( Result ~ having_IPhaving_IP_Address + Shortining_Service + Prefix_Suffix + 
                        having_Sub_Domain + SSLfinal_State + Domain_registeration_length + 
                        Favicon + port + HTTPS_token + Request_URL + URL_of_Anchor + 
                        Links_in_tags + SFH + Redirect + DNSRecord + web_traffic + 
                        Page_Rank + Google_Index + Links_pointing_to_page)

model_new <- glm(myform,family = binomial(link = 'logit'), data= train)

test_predict <- predict(model_new,newdata = test[,-31])
test_predict[test_predict < 0.5] <- 0
test_predict[test_predict > 0.5] <- 1

confusionMatrix(as.factor(test_predict),test[,31]) # Acc : 93.67 and kappa: 87.2%


#plot ROC 
rocplot(model_new)
rocplot(model)
#---------------------------------------------------------------------------------------------------

