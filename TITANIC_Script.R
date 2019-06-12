#################################################################################################
#                                       TITANIC DATASET                                         #
#################################################################################################

################################# Load Library & Dataset #######################################
library(caret)
library(tidyverse)
library(magrittr)
library(rpart)
library(rpart.plot)
library(randomForest)


#Please download dataset from my GitHub : https://github.com/iambatty/Titanic-Project/blob/master/train.csv
df <- read_csv("train.csv")

################################# Data Exploration ############################################

#check data structure
str(df)

#check how many survival
table(df$Survived)
prop.table(table(df$Survived))

#check passengers' gender and survival rate of each gender
table(df$Sex)
prop.table(table(df$Sex, df$Survived))
prop.table(table(df$Sex, df$Survived),1)

#check passengers' age distribution
hist(df$Age)

#check passengers' Fare
hist(df$Fare)

########################## Data Cleansing ###################################################
any(is.na(df$Age))

avg_Age <- mean(df$Age, na.rm = TRUE)

df$Age[is.na(df$Age)] <- avg_Age

############################# Create Train Set - Test Set ###################################

y <- df$Survived

set.seed(1)
test_index <- createDataPartition(y, times = 1, p = 0.2, list = FALSE)

train_set <- df[-test_index,]
test_set <- df[test_index,]

#check survival on train_set
prop.table(table(train_set$Survived))

#check survival on test_set
prop.table(table(test_set$Survived))

########################### The Gender-Class Model #########################################
# Predict that only Female will survive

test_set$y_hat <- 0

test_set$y_hat[test_set$Sex == "female"] <- 1

model_1_confusionMatrix <- table(test_set$Survived, test_set$y_hat)
model_1_accuracy <- mean(test_set$Survived == test_set$y_hat)

accuracy_results <- data.frame(method = "The Gender-Class Model", Accuracy = model_1_accuracy)
accuracy_results %>% knitr::kable()

test_set$y_hat <- NULL
######################## Decision Tree Model ##############################################
#Set model parameters
control <- rpart.control(minsplit = 6,
                         minbucket = round(5 / 3),
                         maxdepth = 5,
                         cp = 0)

fit_dt <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare, 
                data = train_set, method = 'class', control = control)

# Plot Decision Tree
rpart.plot(fit_dt, extra = 106)

test_set$y_hat <- predict(fit_dt, test_set, type = 'class')

model_2_confusionMatrix <- table(test_set$Survived, test_set$y_hat)
model_2_accuracy <- mean(test_set$Survived == test_set$y_hat)

accuracy_results <- bind_rows(accuracy_results,
                          data_frame(method = "Decision Tree",
                                     Accuracy = model_2_accuracy))
accuracy_results %>% knitr::kable()

test_set$y_hat <- NULL
######################## Random Forest Model ##############################################
train_set <- train_set[,c('Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare')]
test_set <- test_set[,c('Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare')]

train_set$Sex <- as.factor(train_set$Sex)
test_set$Sex <- as.factor(test_set$Sex)

set.seed(1)
fit_rf <- randomForest(as.factor(Survived) ~  Pclass + Sex + Age + SibSp + Parch + Fare,
                    data=train_set, 
                    importance=TRUE, 
                    ntree=2000,
                    nodesize = 3,
                    maxnodes = 25)

#plot feature important
varImpPlot(fit_rf)

test_set$y_hat <- predict(fit_rf, test_set)

model_3_confusionMatrix <- table(test_set$Survived, test_set$y_hat)
model_3_accuracy <- mean(test_set$Survived == test_set$y_hat)

accuracy_results <- bind_rows(accuracy_results,
                              data_frame(method = "Random Forest",
                                         Accuracy = model_3_accuracy))

####################### Model Performance Summary #########################################

accuracy_results %>% knitr::kable()