# 2022.06.14

dataset_org <- read.csv("~/Google Drive/Future/Courses/kaggle/Credit-Risk/data/credit_risk_dataset.csv")

library(tidyr)
library(tidyverse)
library(ggplot2)
library(caret)


dataset_longer  <- gather(dataset_org)

dataset_numeric_longer <- dataset_org %>%
  select_if(is.numeric) %>%
  gather()
  
#### Visualising the distributions of the numeric variables ####

ggplot(dataset_numeric_longer) +
  geom_histogram(aes(x = value)) +
  facet_wrap(~key, scales = "free")

#: The distributions are skewed


#### Counting the number of NA values ####

colSums(is.na(dataset_org))

#: Variables person_emp_length (895) and loan_int_rate (3116) have NA values

## To start with let's remove the columns that have NA values

dataset <- dataset_org %>%
  na.omit()

#### Dividing the dataset intro training and test dataset ####

set.seed(1)

trainIndex <- createDataPartition(dataset$cb_person_default_on_file, p = 0.8,
                                  list = FALSE)

train_data <- dataset[trainIndex,]
test_data <- dataset[-trainIndex,]


#### Random forest model on the train_data ####

model_rf <- train(cb_person_default_on_file ~., data = train_data, method = "rf")

## Accuracy of the random forest model ####

confusionMatrix(model_rf)

#: The accuracy on train dataset is 0.8249

#### Prediction on the train dataset
predict_test_data <- predict(object = model_rf, newdata = test_data)

acc_test_rf <- confusionMatrix(data = predict_test_data, reference = as.factor(test_data$cb_person_default_on_file))


#: The accuracy on test dataset is 0.8285


#### Boosted Logistic regression ####

model_blr <- train(cb_person_default_on_file ~., data = train_data, method = "LogitBoost")

predict_test_data_blr <- predict(object = model_blr, newdata = test_data)

acc_test_blr <- confusionMatrix(data = predict_test_data_blr, reference = as.factor(test_data$cb_person_default_on_file))


#### Handling NA values ####

#: Variables person_emp_length and loan_int_rate have NA values

dataset_org_train <- dataset_org[trainIndex,]
dataset_org_test <- dataset_org[-trainIndex,]

model_impute <- preProcess(x = dataset_org_train, method = "knnImpute")

dataset_org_train_impute <- predict(object = model_impute, newdata = dataset_org_train)

model_blr_NA <- train(cb_person_default_on_file ~., data = dataset_org_train_impute, method = "LogitBoost")

dataset_org_test_preprocess <- predict(object = model_impute, newdata = dataset_org_test)

pred_test_data_blr_NA <- predict(object = model_blr_NA, newdata = dataset_org_test_preprocess)


acc_test_blr_NA <- confusionMatrix(data = pred_test_data_blr_NA, reference = as.factor(dataset_org_test$cb_person_default_on_file))

#: Imputing NA values with KNN increased the accuracy from 0.8181 to 0.8268


