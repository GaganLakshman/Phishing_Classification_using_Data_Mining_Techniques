data <- read.csv("dataset.csv")
data <- data[,-c(1,22)]

#Converting the Categorical Variable into Factor
data$Result <- factor(data$Result)

#Splitting the dataset into training data and testing data
#library(caTools)
set.seed(1337)
split <- sample.split(data$Result,SplitRatio = .70)
training <- subset(data,split == TRUE)
test <- subset(data,split==FALSE)


boruta.train <- Boruta::Boruta(Result ~.,data= data,doTrace=2)
print(boruta.train)

plot(boruta.train, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)
  boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
       at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)

library(Boruta)
final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)
getSelectedAttributes(final.boruta, withTentative = F)

