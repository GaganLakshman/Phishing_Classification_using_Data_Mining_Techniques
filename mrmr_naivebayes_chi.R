library(caTools)
library(foreach)
library(arules)
library(plyr)
library(arulesViz)
library(dummies)
library(caret)
library(dplyr)
library(mRMRe)
data <- read.csv("new_dataset.csv", colClasses = "factor")

data <- data[,-1]
#FEATURE SELECTION BASED ON CHI AND IG
#####################
#FEATURE SELECTION BASED ON INFORMATION GAIN

library(FSelector)
#information gain of an attribute tells you how much information with respect to the classification target the attribute gives you. 
#That is, it measures the difference in information between the cases where you know the value of the attribute and where you don't know the value of the attribute.
#morethe score better is the variable
weights <- information.gain(Result~., data)
print(weights)
subset <- cutoff.k(weights, 30)

#FEATURE SELECTION BASED ON CHI SQUARE STATISTIC

library(FSelector)
library(ggplot2)

#Finds the importance of attributes based on chi square value, more the value better is the attribute
weights_chi <- chi.squared(Result~.,data = data)
print(weights_chi)
subset_chi <- cutoff.k(weights_chi,30)

#RANKING FEATURES BY COMBINING INFORMATION GAIN AND CHI SQUARE STATISTICS.(BASED ON PAPER WRIITEN BY KHIRAN D RAJAB)
#based on the paper written by KHIRAN D RAJAB,we can combine chi square value and information gain value to rank the variables 
#to combine the data it is important to normalize the data so that both the values are at the same level
IGMax <- max(weights)
weights_IG_normalized <- weights/IGMax 
#each chisquare value has been squared 
sqr_weights_IG_normalized <- as.list(apply(weights_IG_normalized,1,function(x) x^2))
sqr_weights_IG_normalized <- unlist(sqr_weights_IG_normalized, use.names=FALSE)

ChiMax <- max(weights_chi)
weights_chi_normalized <- weights_chi/ChiMax
#each information gain value has been squared 
sqr_weights_chi_normalized <- as.list(apply(weights_chi_normalized,1,function(x) x^2))
sqr_weights_chi_normalized <- unlist(sqr_weights_chi_normalized, use.names=FALSE)
#according to the formula we need to sum the squares and take the square root of it to get the metrics.
rank_sum <- sqr_weights_chi_normalized+sqr_weights_IG_normalized
rank_combining_IG_Chi <- sqrt(rank_sum)
#formed a table with new values and the feature name
colnames <- rownames(weights_chi_normalized)
rank <- as.data.frame(cbind(colnames,rank_combining_IG_Chi))
#converted factor tonumeric for plotting the graph
as.numeric.factor <- function(x) {as.numeric(levels(x))[x]}
rank$rank_combining_IG_Chi <- as.numeric.factor(rank$rank_combining_IG_Chi)
rank$colnames <- as.character(rank$colnames)
#sorted the table in descending order of the value 
rank <- rank[order(rank$rank_combining_IG_Chi,decreasing = T),]
rank_top10 <- rank[1:10,]
#graph is plotted
p2 <- ggplot(rank_top10, aes(x = reorder(rank_top10$colnames,-rank_top10$rank_combining_IG_Chi), y = rank_top10$rank_combining_IG_Chi)) +
  geom_bar(stat = "identity")+ggtitle("Variable importance based on chisquare and Information Gain")+xlab("Features")+ylab("Score")
require("gridExtra")
grid.arrange(arrangeGrob(p2))

#to get complete data 
data_new <- data[,rank_colnames]
result_data <- data$Result
naive_data <- cbind(data_new,result_data)
  
#FEATURE SELECTION BASED ON MRMRE
dat <- read.csv("new_dataset.csv",colClasses = "factor")
#MRMR accepts eithernumeric or ordered factor, so variables had to be converted into factors 
dat$having_IPhaving_IP_Address <- factor(data$having_IPhaving_IP_Address,ordered = TRUE)
dat$URLURL_Length <- factor(data$URLURL_Length,ordered = TRUE)
dat$Shortining_Service <- factor(data$Shortining_Service,ordered = TRUE)
dat$having_At_Symbol <- factor(data$having_At_Symbol,ordered = TRUE)
dat$double_slash_redirecting <- factor(data$double_slash_redirecting,ordered = TRUE)
dat$Prefix_Suffix <- factor(data$Prefix_Suffix,ordered = TRUE)
dat$having_Sub_Domain <- factor(data$having_Sub_Domain,ordered = TRUE)
dat$SSLfinal_State <- factor(data$SSLfinal_State,ordered = TRUE)
dat$Domain_registeration_length <- factor(data$Domain_registeration_length,ordered = TRUE)
dat$Favicon <- factor(data$Favicon,ordered = TRUE)
dat$port <- factor(data$port,ordered = TRUE)
dat$HTTPS_token <- factor(data$HTTPS_token,ordered = TRUE)
dat$Request_URL <- factor(data$Request_URL,ordered = TRUE)
dat$URL_of_Anchor <- factor(data$URL_of_Anchor,ordered = TRUE)
dat$Links_in_tags <- factor(data$Links_in_tags,ordered = TRUE)
dat$SFH <- factor(data$SFH,ordered = TRUE)
dat$Submitting_to_email <- factor(data$Submitting_to_email,ordered = TRUE)
dat$Abnormal_URL <- factor(data$Abnormal_URL,ordered = TRUE)
dat$Redirect <- factor(data$Redirect,ordered = TRUE)
dat$on_mouseover <- factor(data$on_mouseover,ordered = TRUE)
dat$RightClick <- factor(data$RightClick,ordered = TRUE)
dat$popUpWidnow <- factor(data$popUpWidnow,ordered = TRUE)
dat$Iframe <- factor(data$Iframe,ordered = TRUE)
dat$age_of_domain <- factor(data$age_of_domain,ordered = TRUE)
dat$DNSRecord <- factor(data$DNSRecord,ordered = TRUE)
dat$web_traffic <- factor(data$web_traffic,ordered = TRUE)
dat$Page_Rank <- factor(data$Page_Rank,ordered = TRUE)
dat$Google_Index <- factor(data$Google_Index,ordered = TRUE)
dat$Links_pointing_to_page <- factor(data$Links_pointing_to_page,ordered = TRUE)
dat$Statistical_report <- factor(data$Statistical_report,ordered = TRUE)
dat$Result <- factor(data$Result,ordered = TRUE)

View(dat)
dat <- dat[,-1]
library(mRMRe)
Mrmr_data <- mRMR.data(dat)
mutual_information_matrix <- print(mim(subsetData(Mrmr_dat)))
fs_classic <- mRMR.classic(data = Mrmr_data, target_indices = 31,feature_count = 10,method = "bootstrap")

solutions(fs_classic)
x <- unlist(solutions(fs_classic))
mrmrdata<-data[,x]
mrmrdata<-cbind(mrmrdata,data$Result) 
write.csv(mrmr_data)

network <- new("mRMRe.Network", data = data1, target_indices = 31,levels = c(2,1), layers =10)
visualize(network)

################################################
## Naive Bayes
# Importing the dataset
dataset = read.csv('mrmrdata.csv')
dataset = dataset[,-1]

# Encoding the target feature as factor
dataset$data.Result = factor(dataset$data.Result)

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(1337)
split = sample.split(dataset$data.Result, SplitRatio = 0.70)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

install.packages('e1071')
library(e1071)
classifier = naiveBayes(x = training_set[,-11],
                        y = training_set$data.Result)
#x is the training set without the output vaiable and y is the depended variable

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[,-11])

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)
#depended variable should be of factor type else it will throw error

##Naive bayes model with 
split = sample.split(data$Result, SplitRatio = 0.70)
training_set_data = subset(data, split == TRUE)
test_set_data = subset(data, split == FALSE)

install.packages('e1071')
library(e1071)
classifier = naiveBayes(x = training_set_data[,-31],
                        y = training_set_data$data.Result)
#x is the training set without the output vaiable and y is the depended variable

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[,-31])

# Making the Confusion Matrix
cm = table(test_set[, 31], y_pred)
#depended variable should be of factor type else it will throw error

##MFA
library(FactoMineR)
library(factoextra)
setwd("C:/Mscdad/ADM")
data <- read.csv("new_dataset.csv",colClasses = "factor")
data <- data[,-1]
data <- data[,-31]
#all the predictors are grouped into 4 groups and no supplementary variables are considered,thus all variables are considered to have equal contribution
res_mfa <-  MFA(data,group = c(13,5,5,7),type =c(rep("n",4)),ncp = 5,name.group = c("AddressBar","Abnormal","HTML","Domain"))
summary(res_mfa)
res_mfa$eig
#This allows us to estimate how well the groups has contributed with the dimensions
res_mfa$group$Lg
#RV is used to find the how close groupsare with each other
res_mfa$group$RV
#eigen values indicates the amount of transformation indictaing how components are created 
eig.val <- get_eigenvalue(res_mfa)
print(eig.val)
fviz_screeplot(res_mfa)

fviz_mfa_var(res_mfa, "group")

# Contribution to the first dimension
fviz_contrib(res_mfa, "group", axes = 1)
# Contribution to the second dimension
fviz_contrib(res_mfa, "group", axes = 2)

#spread of groups along dimensions 1 and 2
fviz_mfa_var(res_mfa, "group", palette = "jco", 
             col.var.sup = "violet", repel = TRUE)

 
