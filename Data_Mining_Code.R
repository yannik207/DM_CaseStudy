#---------------------------------------------------------------------------------------------------------------------------------------
# Projekt Betrug 
#---------------------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------------------
# Workspace preparation
#---------------------------------------------------------------------------------------------------------------------------------------

# Load packages 
library(corrplot)
library(corrgram)
library(readr)
library(ggplot2)
library(tidyverse)
library(rpart)
library(data.table)
library(smbinning)
library(rpart.plot)
library(ROCR)
library(dplyr)
library(randomForest)
library(foreach)
library(doParallel)
library(ada)
library(gbm)
library(xgboost)

#---------------------------------------------------------------------------------------------------------------------------------------
# Data import 
#---------------------------------------------------------------------------------------------------------------------------------------

# import data set 
self_checkout_data <- read_csv("~/Desktop/BWL-Studium/BWL 5 Semester/QM 2/Data Mining/DM_assignment/R/self_checkout_data.csv")
self_checkout_data_org <- read_csv("~/Desktop/BWL-Studium/BWL 5 Semester/QM 2/Data Mining/DM_assignment/R/self_checkout_data.csv")

# create data table
self_checkout_data <- as.data.table(self_checkout_data)

# import data set without label 
test_ohne_zv <- read_csv("~/Desktop/BWL-Studium/BWL 5 Semester/QM 2/Data Mining/DM_assignment/R/self_checkout_scoring.csv")

#---------------------------------------------------------------------------------------------------------------------------------------
# Data inspection (1) + Data visualization (1)
#---------------------------------------------------------------------------------------------------------------------------------------

# how many NA's are in the dataset relative to the number of observations?
org_nrow <- nrow(self_checkout_data)
sum(is.na(self_checkout_data))/nrow(self_checkout_data)

# overview
str(self_checkout_data)
summary(self_checkout_data)

# proportion of fraudulent cases 
perc_fraud <- table(self_checkout_data$fraud)[2]/nrow(self_checkout_data)
perc_fraud

# how much do we currently lose due to fraud? 
fraud_data <- self_checkout_data[self_checkout_data$fraud == 1, ]
fraud_data

total_loss_fraud <- sum(nrow(fraud_data)*-5)
total_loss_fraud

# The following shows the difference between the fraudulent and the not fraudulent group regarding the considered variable
# Generally we expect if the two considered variables are significantly different from one another it might indicate that this variable is able to predict fraud.

# trust level
table(self_checkout_data$trustLevel, self_checkout_data$fraud)

# totalScanTimeInSeconds
boxplot(self_checkout_data$totalScanTimeInSeconds ~ self_checkout_data$fraud, 
        main = "Distribution", xlab = "fraud", ylab = "totalScanTimeInSeconds", col=topo.colors(2))
legend("bottom", inset=.02,
       c("not fraudulent (0)","fraudulent (1)"), fill=topo.colors(2), horiz=TRUE, cex=0.8)
t.test(self_checkout_data$totalScanTimeInSeconds ~ self_checkout_data$fraud)

# grandTotal
boxplot(self_checkout_data$grandTotal ~ self_checkout_data$fraud, 
        main = "Distribution", xlab = "fraud", ylab = "grandTotal", col=topo.colors(2))
legend("bottom", inset=.02,
       c("not fraudulent (0)","fraudulent (1)"), fill=topo.colors(2), horiz=TRUE, cex=0.8)
t.test(self_checkout_data$grandTotal ~ self_checkout_data$fraud)

# lineItemVoids
boxplot(self_checkout_data$lineItemVoids ~ self_checkout_data$fraud, 
        main = "Distribution", xlab = "fraud", ylab = "lineItemVoids", col=topo.colors(2))
legend("bottom", inset=.02,
       c("not fraudulent (0)","fraudulent (1)"), fill=topo.colors(2), horiz=TRUE, cex=0.8)
t.test(self_checkout_data$lineItemVoids ~ self_checkout_data$fraud)

# scansWithoutRegistration
boxplot(self_checkout_data$scansWithoutRegistration ~ self_checkout_data$fraud,
        main = "Distribution", xlab = "fraud", ylab = "scansWithoutRegistration", col=topo.colors(2))
legend("bottom", inset=.02,
       c("not fraudulent (0)","fraudulent (1)"), fill=topo.colors(2), horiz=TRUE, cex=0.8)
t.test(self_checkout_data$scansWithoutRegistration ~ self_checkout_data$fraud)

# quantityModifications
boxplot(self_checkout_data$quantityModifications ~ self_checkout_data$fraud,
        main = "Distribution", xlab = "fraud", ylab = "quantityModifications", col=topo.colors(2))
legend("bottom", inset=.02,
       c("not fraudulent (0)","fraudulent (1)"), fill=topo.colors(2), horiz=TRUE, cex=0.8)
t.test(self_checkout_data$quantityModifications ~ self_checkout_data$fraud)
# --> quantityModifications has no significant difference in these two groups

# scannedLineItemsPerSecond
boxplot(self_checkout_data$scannedLineItemsPerSecond ~ self_checkout_data$fraud,
        main = "Distribution", xlab = "fraud", ylab = "scannedLineItemsPerSecond", col=topo.colors(2))
legend("bottom", inset=.02,
       c("not fraudulent (0)","fraudulent (1)"), fill=topo.colors(2), horiz=TRUE, cex=0.8)
t.test(self_checkout_data$scannedLineItemsPerSecond ~ self_checkout_data$fraud)
# --> There are outliers which we can not explain since no human being is able to scan 30 items per second. Our assumption is that these are tracking errors and hence must be cleaned.

# valuePerSecond
boxplot(self_checkout_data$valuePerSecond ~ self_checkout_data$fraud,
        main = "Distribution", xlab = "fraud", ylab = "valuePerSecond", col=topo.colors(2))
legend("bottom", inset=.02,
       c("not fraudulent (0)","fraudulent (1)"), fill=topo.colors(2), horiz=TRUE, cex=0.8)
t.test(self_checkout_data$valuePerSecond ~ self_checkout_data$fraud)

# lineItemVoidsPerPosition
boxplot(self_checkout_data$lineItemVoidsPerPosition ~ self_checkout_data$fraud,
        main = "Distribution", xlab = "fraud", ylab = "lineItemVoidsPerPosition", col=topo.colors(2))
legend("bottom", inset=.02,
       c("not fraudulent (0)","fraudulent (1)"), fill=topo.colors(2), horiz=TRUE, cex=0.8)
t.test(self_checkout_data$lineItemVoidsPerPosition ~ self_checkout_data$fraud)

# correlation matrix
corrgram(self_checkout_data)

# distribution for each variable
self_checkout_data %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

#---------------------------------------------------------------------------------------------------------------------------------------
# Selection
#---------------------------------------------------------------------------------------------------------------------------------------
# we select all data 
#---------------------------------------------------------------------------------------------------------------------------------------
# Cleansing (2)
#---------------------------------------------------------------------------------------------------------------------------------------

# drop checkoutID because it does not contain any useful information 
self_checkout_data$checkoutID <- NULL 

# drop irrelevant (relevance based on the t.test) variable - quantityModifications
self_checkout_data$quantityModifications <- NULL 

# drop every column where scannedLineItemsPerSecond > 5, because it's quite unrealistic that you're able to scan more than 5 items per second
self_checkout_data <- self_checkout_data[self_checkout_data$scannedLineItemsPerSecond <= 5, ]

# hom many data points did we lose? (in percentage)
(1-(nrow(self_checkout_data)/org_nrow))*100

# Binning
# Binning of variables (also called optimal binning) - create new categorical variable, where categories are as heterogeneous as possible
binning_result_1 <- smbinning(as.data.frame(self_checkout_data), y = "fraud", x = "totalScanTimeInSeconds", p = 0.05)
self_checkout_data <- smbinning.gen(self_checkout_data, binning_result_1, "SessionTimeCat1")

binning_result_2 <- smbinning(as.data.frame(self_checkout_data), y = "fraud", x = "grandTotal", p = 0.05)
self_checkout_data <- smbinning.gen(self_checkout_data, binning_result_2, "SessionTimeCat2")

binning_result_3 <- smbinning(as.data.frame(self_checkout_data), y = "fraud", x = "lineItemVoids", p = 0.05)
self_checkout_data <- smbinning.gen(self_checkout_data, binning_result_3, "SessionTimeCat3")

## New interesting variables 

# total invalid scans - all scans that are not valid either because of voided scans or attempts that were not successful (scansWithoutRegistration)
self_checkout_data$totalInvalidScans <- self_checkout_data$lineItemVoids + self_checkout_data$scansWithoutRegistration
boxplot(self_checkout_data$totalInvalidScans ~ self_checkout_data$fraud, 
        main = "Distribution", xlab = "fraud", ylab = "totalInvalidScans",col=topo.colors(2))
legend("bottom", inset=.02,
       c("not fraudulent (0)","fraudulent (1)"), fill=topo.colors(2), horiz=TRUE, cex=0.8)

# Does valuePerItem make a difference in having a fraudulent/ non fraudulent customer? 
self_checkout_data <- mutate(self_checkout_data, valuePerItem = grandTotal/(totalScanTimeInSeconds*scannedLineItemsPerSecond))
boxplot(self_checkout_data$valuePerItem ~ self_checkout_data$fraud, 
        main = "Distribution", xlab = "fraud", ylab = "valuePerItem",col=topo.colors(2))
legend("bottom", inset=.02,
       c("not fraudulent (0)","fraudulent (1)"), fill=topo.colors(2), horiz=TRUE, cex=0.8)

# convert fraud into factor
self_checkout_data$fraud <- as.factor(self_checkout_data$fraud)
#convert trustLevel into factor
self_checkout_data$trustLevel <- as.factor(self_checkout_data$trustLevel)

#---------------------------------------------------------------------------------------------------------------------------------------
# Construction (3)
#---------------------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------------------
# Generate train and test data
#---------------------------------------------------------------------------------------------------------------------------------------

set.seed(123)

# Define training and test data sets
train_share <- 0.7
sample_size <- nrow(self_checkout_data)
sample_train <- sample(x = 1:sample_size, size = floor(sample_size*train_share), 
                       replace = FALSE)
sample_test <- setdiff(1:sample_size, sample_train)

# make matrix a df 
self_checkout_data <- as.data.frame(self_checkout_data)

df_train <- self_checkout_data[sample_train,]
df_test <- self_checkout_data[sample_test,]

df_train_x <- df_train[,-which(names(df_train) %in% "fraud")]
df_train_y <- factor(df_train[,"fraud"])
df_test_x <- df_test[,-which(names(df_test) %in% "fraud")]
df_test_y <- factor(df_test[,"fraud"])

#---------------------------------------------------------------------------------------------------------------------------------------
# Train Decision Tree
#---------------------------------------------------------------------------------------------------------------------------------------

# Train model based on training data only
set.seed(123)
dt_model <- rpart(formula = fraud ~ .,
                  data = df_train, 
                  method = "class", control = rpart.control(cp = 0.000130998))

# print error
printcp(dt_model)

# plot model
plotcp(dt_model)

## Predict 

# Predict training data set
# classification - TRUE or FALSE - default predict threshold is 50% 
pred_train_class <- predict(dt_model, newdata = df_train, type = "class")
# probability 
pred_train <- predict(dt_model, newdata = df_train, type = "prob")

# Predict test data set
# classification 
pred_test_class <- predict(dt_model, newdata = df_test, type = "class")
# probability 
pred_test <- predict(dt_model, newdata = df_test, type = "prob")

# Compute confusion matrices
cm_train <- table(Prediction = pred_train_class, 
                  Reality = df_train$fraud)
cm_test <- table(Prediction = pred_test_class, 
                 Reality = df_test$fraud)

# Compute accuracy - might not be very meaningful, because it also includes predictions that might be very easy to call
sum(diag(cm_train))/sum(cm_train)
sum(diag(cm_test))/sum(cm_test)

#---------------------------------------------------------------------------------------------------------------------------------------
# Performance measurement (4)
#---------------------------------------------------------------------------------------------------------------------------------------

# Compute prediction object

# prediction - prediction and truth are being put into the function
prediction_train <- prediction(pred_train[,2], labels = df_train$fraud)
prediction_test <- prediction(pred_test[,2], labels = df_test$fraud)

# Plot ROC
roc_train <- performance(prediction_train, measure="tpr", x.measure="fpr")
roc_test <- performance(prediction_test, measure="tpr", x.measure="fpr")

plot(roc_train)
plot(roc_test, add = TRUE, col = "red")
abline(0,1, col = "grey", lty = 2)
legend("topright", legend = c("Train", "Test"), col = c("black", "red"), lty = c(1,1))

# Lift
lift_train <- performance(prediction_train, measure="lift", x.measure="rpp")
lift_test <- performance(prediction_test, measure="lift", x.measure="rpp")

plot(lift_train)
plot(lift_test, add = TRUE, col = "red")
abline(h = 1, col = "black", lty = 2)
legend("topright", legend = c("Train", "Test"), col = c("black", "red"), lty = c(1,1))

#---------------------------------------------------------------------------------------------------------------------------------------
# Random Forest 
#---------------------------------------------------------------------------------------------------------------------------------------

# Train model
set.seed(123)
rf_model <- randomForest(x = df_train_x, y = df_train_y, 
                         xtest = df_test_x, ytest = df_test_y, 
                         do.trace = 20, importance = TRUE)
# No. of variables tried at each split: 3 - Anzahl der Variablen die im jedem Split ausprobiert wurden - Ist die Wurzel aus der Anzahl der Variablen 
rf_model

# Tune mtry - Parameter Tuning
tune_mtry <- tuneRF(x = df_train_x, 
                    y = df_train_y, 
                    mtryStart = 3)

# Build tuned model
set.seed(123)
rf_model <- randomForest(x = df_train_x, y = df_train_y, 
                         xtest = df_test_x, ytest = df_test_y, 
                         mtry = 6, do.trace = 20, importance = TRUE, 
                         localImp = TRUE, ntree = 220,
                         keep.forest = TRUE)
# --> the model is not significantly improving after ntree=220

# plot model
plot(rf_model)

#---------------------------------------------------------------------------------------------------------------------------------------
# Predict
#---------------------------------------------------------------------------------------------------------------------------------------

# Predict training data set
pred_train_class <- rf_model$predicted
pred_train <- rf_model$votes

# Predict test data set
pred_test_class <- rf_model$test$predicted
pred_test <- rf_model$test$votes

# Compute confusion matrices
cm_train <- table(pred_train_class, df_train$fraud)
cm_test <- table(pred_test_class, df_test$fraud)

# Compute accuracy
sum(diag(cm_train))/sum(cm_train)
sum(diag(cm_test))/sum(cm_test)

# check for overfit 

# Compute prediction object
prediction_train <- prediction(pred_train[,2], labels = df_train$fraud)
prediction_test <- prediction(pred_test[,2], labels = df_test$fraud)

# Plot ROC 
roc_train <- performance(prediction_train, measure="tpr", x.measure="fpr")
roc_test <- performance(prediction_test, measure="tpr", x.measure="fpr")

plot(roc_train)
plot(roc_test, add = TRUE, col = "red")
abline(0,1, col = "grey", lty = 2)
legend("topright", legend = c("Train", "Test"), col = c("black", "red"), lty = c(1,1))

#---------------------------------------------------------------------------------------------------------------------------------------
# Boosting 
#---------------------------------------------------------------------------------------------------------------------------------------

# Train ada Model
set.seed(123)
ada_model <- ada(x = df_train_x, y = df_train_y,
                 test.x = df_test_x, test.y = df_test_y, 
                 loss = "exponential", type = "discrete", iter = 100)

ada_model
plot(ada_model, test = TRUE, xlab = "Iteration", ylab = "Error")

# GBM has issues with factor var as the target variable 
df_train$fraud <- ifelse(df_train$fraud == 1, 1, 0)

# Train GBM model
set.seed(123)
gbm_model <- gbm(formula = fraud ~ ., data = df_train, interaction.depth = 10,
                 n.trees = 1000, distribution = "bernoulli")
# not significant increase overall after 200, but specifically the alpha error is improving this is why we used n.trees=1000

# plot train error
plot(gbm_model$train.error, xlab = "number of trees", ylab = "Error", type = "l", main = "Training Error")

# convert into factor again
df_train$fraud <- as.factor(df_train$fraud)

# Train xgboost model
# create xgb train and test data set
xgb_df_train <- xgb.DMatrix(model.matrix(object = fraud ~ ., data = df_train), label = ifelse(df_train_y==1, 1, 0))
xgb_df_test <- xgb.DMatrix(model.matrix(object = fraud ~ ., data = df_test), label = ifelse(df_test_y==1, 1, 0))
param <- list(max_depth = 6, eta = 0.3, nthread = 2, 
              objective = "binary:logistic", eval_metric = "auc")
# previously created xgb objects are being placed into the xgb.train() function 
set.seed(123)
xgb_model <- xgb.train(param, xgb_df_train,nrounds = 500)

# Predict test data set
pred_test_dt <- predict(dt_model, newdata = df_test_x, type = "prob")
pred_test_rf <- rf_model$test$votes
pred_test_ada <- predict(ada_model, newdata = df_test_x, type = "probs")
pred_test_gbm <- predict(gbm_model, n.trees = gbm_model$n.trees, newdata = df_test_x, type = "response")
pred_test_xgb <- predict(xgb_model, newdata = xgb_df_test)

#---------------------------------------------------------------------------------------------------------------------------------------
# Performance measurement (4)
#---------------------------------------------------------------------------------------------------------------------------------------

# Compute prediction object
library(ROCR)
prediction_test_dt <- prediction(pred_test_dt[,2], labels = df_test$fraud)
prediction_test_rf <- prediction(pred_test_rf[,2], labels = df_test$fraud)
prediction_test_ada <- prediction(pred_test_ada[,2], labels = df_test$fraud)
prediction_test_gbm <- prediction(pred_test_gbm, labels = df_test$fraud)
prediction_test_xgb <- prediction(pred_test_xgb, labels = df_test$fraud)

# Plot ROC 
roc_test_dt <- ROCR::performance(prediction_test_dt, measure="tpr", x.measure="fpr")
roc_test_rf <- ROCR::performance(prediction_test_rf, measure="tpr", x.measure="fpr")
roc_test_ada <- ROCR::performance(prediction_test_ada, measure="tpr", x.measure="fpr")
roc_test_gbm <- ROCR::performance(prediction_test_gbm, measure="tpr", x.measure="fpr")
roc_test_xgb <- ROCR::performance(prediction_test_xgb, measure="tpr", x.measure="fpr")

# Compute AUC
auc_test_dt <- ROCR::performance(prediction_test_dt, measure="auc")
auc_test_rf <- ROCR::performance(prediction_test_rf, measure="auc")
auc_test_ada <- ROCR::performance(prediction_test_ada, measure="auc")
auc_test_gbm <- ROCR::performance(prediction_test_gbm, measure="auc")
auc_test_xgb <- ROCR::performance(prediction_test_xgb, measure="auc")

auc_test_dt <- round(auc_test_dt@y.values[[1]], 4)
auc_test_rf <- round(auc_test_rf@y.values[[1]], 4)
auc_test_ada <- round(auc_test_ada@y.values[[1]], 4)
auc_test_gbm <- round(auc_test_gbm@y.values[[1]], 4)
auc_test_xgb <- round(auc_test_xgb@y.values[[1]], 4)

# plot AUC for all models
plot(roc_test_dt)
plot(roc_test_rf, add = TRUE, col = "red")
plot(roc_test_ada, add = TRUE, col = "blue")
plot(roc_test_gbm, add = TRUE, col = "green")
plot(roc_test_xgb, add = TRUE, col = "orange")
abline(0,1, col = "grey", lty = 2)
legend("bottomright", legend = c(paste0("Decision Tree ( AUC = ", auc_test_dt, ")"),
                                 paste0("RandomForest ( AUC = ", auc_test_rf, ")"),
                                 paste0("AdaBoost ( AUC = ", auc_test_ada, ")"),
                                 paste0("GBM ( AUC = ", auc_test_gbm, ")"),
                                 paste0("XGBoost ( AUC = ", auc_test_xgb, ")")),
       col = c("black", "red", "blue", "green", "orange"), lty = c(1,1,1,1,1))

#---------------------------------------------------------------------------------------------------------------------------------------
# Cost optimal cutoffs (4)
#---------------------------------------------------------------------------------------------------------------------------------------

# Define cost matrix (as in instructions) decision tree
costs <- matrix(c(0,-25,-5,5), ncol = 2, byrow = TRUE)
rownames(costs) <- colnames(costs) <- c(FALSE, TRUE)
costs <- as.table(costs)
names(dimnames(costs)) <- c("Reality", "Prediction")
costs

# different threshold - roughly at 0.79 through the threshold computation 
cm2_dt <- table(df_test$fraud, pred_test_dt[,2]>0.79, dnn = c("Reality", "Prediction"))
cm2_dt

cost_matrx_dt <- sum(cm2_dt*costs)
cost_matrx_dt

# Function to compute costs for any given threshold
costf <- function(x){
  cmx <- table(df_test$fraud, pred_test_dt[,2]>x, dnn = c("Reality", "Prediction"))
  return(sum(cmx*costs))
}

# Compute costs for various thresholds
s <- seq(0.05,0.9,0.01)
r <- numeric(0)
for (i in s){
  r <- c(r,costf(i))
}
plot(s,r,type = "l", xlab = "Threshold", ylab = "Total Margin")
s[which.max(r)]

#---------------------------------------------------------------------------------------------------------------------------------------

# Compute costs or standard prediction
cm2_rf <- table(df_test$fraud, pred_test[,2]>0.56, dnn = c("Reality", "Prediction"))
cm2_rf

cost_matrx_rf <- sum(cm2_rf*costs)
cost_matrx_rf

# Function to compute costs for any given threshold
costf <- function(x){
  cmx <- table(df_test$fraud, pred_test[,2]>x, dnn = c("Reality", "Prediction"))
  return(sum(cmx*costs))
}

# Compute costs for various thresholds
s <- seq(0.01,0.9,0.01)
r <- numeric(0)
for (i in s){
  r <- c(r,costf(i))
}
plot(s,r,type = "l", xlab = "Threshold", ylab = "Total Margin")
s[which.max(r)]

#---------------------------------------------------------------------------------------------------------------------------------------

# Compute costs or standard prediction
cm2_ada <- table(df_test$fraud, pred_test_ada[,2]>0.57, dnn = c("Reality", "Prediction"))
cm2_ada

# cost matrix ada 
cost_matrx_ada <- sum(cm2_ada*costs)
cost_matrx_ada

# Function to compute costs for any given threshold
costf <- function(x){
  cmx <- table(df_test$fraud, pred_test_ada[,2]>x, dnn = c("Reality", "Prediction"))
  return(sum(cmx*costs))
}

# Compute costs for various thresholds
s <- seq(0.01,0.9,0.01)
r <- numeric(0)
for (i in s){
  r <- c(r,costf(i))
}
plot(s,r,type = "l", xlab = "Threshold", ylab = "Total Margin")
s[which.max(r)]

#---------------------------------------------------------------------------------------------------------------------------------------

# Compute costs or standard prediction
cm2_gbm <- table(df_test$fraud, pred_test_gbm>0.52, dnn = c("Reality", "Prediction"))
cm2_gbm

# cost matrix gbm 
cost_matrx_gbm <- sum(cm2_gbm*costs)
cost_matrx_gbm

# Function to compute costs for any given threshold
costf <- function(x){
  cmx <- table(df_test$fraud, pred_test_gbm>x, dnn = c("Reality", "Prediction"))
  return(sum(cmx*costs))
}

# Compute costs for various thresholds
s <- seq(0.01,0.9,0.01)
r <- numeric(0)
for (i in s){
  r <- c(r,costf(i))
}
plot(s,r,type = "l", xlab = "Threshold", ylab = "Total Margin")
s[which.max(r)]

#---------------------------------------------------------------------------------------------------------------------------------------

# Compute costs or standard prediction
cm2_XGBoost <- table(df_test$fraud, pred_test_xgb>0.52, dnn = c("Reality", "Prediction"))
cm2_XGBoost

# cost matrix gbm
cost_matrx_XGBoost <- sum(cm2_XGBoost*costs)
cost_matrx_XGBoost

# Function to compute costs for any given threshold
costf <- function(x){
  cmx <- table(df_test$fraud, pred_test_gbm>x, dnn = c("Reality", "Prediction"))
  return(sum(cmx*costs))
}

# Compute costs for various thresholds
s <- seq(0.01,0.9,0.01)
r <- numeric(0)
for (i in s){
  r <- c(r,costf(i))
}
plot(s,r,type = "l", xlab = "Threshold", ylab = "Total Margin")
s[which.max(r)]

#---------------------------------------------------------------------------------------------------------------------------------------
# Variable Importance
#---------------------------------------------------------------------------------------------------------------------------------------

# decision tree
# Visualize variable importance
top_var_dt <- dt_model$variable.importance[dt_model$variable.importance > mean(dt_model$variable.importance)]
barplot(top_var_dt, horiz = FALSE, col=topo.colors(50), main = "Variable Importance Decision Tree Model", cex.names=0.9, ylab = "Score", las = 3)

# random forest 
# Visualize variable importance
varImpPlot(rf_model)

# ada 
# Visualize variable importance 
varplot(ada_model)

# gbm 
# Visualize variable importance
summary_gbm <- as.data.table(summary(gbm_model))
top_var_gbm <- summary_gbm[summary_gbm$rel.inf > mean(summary_gbm$rel.inf)]
barplot(top_var_gbm$rel.inf, names.arg = top_var_gbm$var, horiz = FALSE, las = 1, col=topo.colors(30), main = "Variable Importance GBM-Model", cex.names=0.9, ylab = "Score", las = 3)

# XGBoost
# Visualize variable importance
xgb_importance_class <- xgb.importance(model = xgb_model)
top_var_xgb <- xgb_importance_class[xgb_importance_class$Gain> mean(xgb_importance_class$Gain)]
barplot(top_var_xgb$Gain, names.arg = top_var_xgb$Feature, horiz = FALSE, col=topo.colors(30), main = "Variable Importance XGBoost-Model", cex.names=0.9, ylab = "Score", las = 3)

#---------------------------------------------------------------------------------------------------------------------------------------
# Save Prediction Results
#---------------------------------------------------------------------------------------------------------------------------------------

# Prediction with unlabeled data set
# data manipulation for XGBoost prediction
test_ohne_zv$fraud <- ifelse(test_ohne_zv$quantityModifications >= 1, 1, 0)

test_ohne_zv$trustLevel <- as.factor(test_ohne_zv$trustLevel)

checkoutID_final <- test_ohne_zv$checkoutID
test_ohne_zv$checkoutID <- NULL

test_ohne_zv$quantityModifications <- NULL

## add previously added "New interesting variables" 
# totalInvalidScans 
test_ohne_zv$totalInvalidScans <- test_ohne_zv$lineItemVoids + test_ohne_zv$scansWithoutRegistration

# valuePerItem
test_ohne_zv <- mutate(test_ohne_zv, valuePerItem = grandTotal/(totalScanTimeInSeconds*scannedLineItemsPerSecond))

# add Binning Variables 
# binning_result_1 <- smbinning(as.data.frame(test_ohne_zv), y = "fraud", x = "totalScanTimeInSeconds", p = 0.05)
test_ohne_zv <- smbinning.gen(test_ohne_zv, binning_result_1, "SessionTimeCat1")

# binning_result_2.1 <- smbinning(as.data.frame(test_ohne_zv), y = "fraud", x = "grandTotal", p = 0.05)
test_ohne_zv <- smbinning.gen(test_ohne_zv, binning_result_2, "SessionTimeCat2")

# binning_result_3.1 <- smbinning(as.data.frame(test_ohne_zv), y = "fraud", x = "lineItemVoids", p = 0.05)
test_ohne_zv <- smbinning.gen(test_ohne_zv, binning_result_3, "SessionTimeCat3")

# convert fraud to factor
test_ohne_zv$fraud <- as.factor(test_ohne_zv$fraud)

# Colname Reihenfolge ändern
col_reihenfolge <- colnames(df_train)
test_ohne_zv <- test_ohne_zv[,col_reihenfolge ]

# test
colnames(df_train) == colnames(test_ohne_zv)
# convert into data table
test_ohne_zv <- as.data.table(test_ohne_zv)

# create prediction with XGBoost
test_ohne_zv_XGB <- xgb.DMatrix(model.matrix(object = fraud ~ ., data = test_ohne_zv))
prediction_unlabeled <-  as.data.table(predict(xgb_model, newdata = test_ohne_zv_XGB))

# add checkoutID_final
prediction_unlabeled$checkoutID <- checkoutID_final
prediction_unlabeled$prediction <- floor(prediction_unlabeled$V1)
prediction_unlabeled$V1 <- NULL

prediction_unlabeled <- as.data.frame(prediction_unlabeled) 

save(prediction_unlabeled,
     file = "prediction_DM.RDATA")