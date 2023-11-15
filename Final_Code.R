#loading required package
rm(list = ls())
set.seed(1)


library(corrplot)
library(tidyverse)
library(ggplot2)
library(rpart.plot)
library(ROSE)
library(kernlab)
library(dplyr)
library(rpart)
library(rpart.plot)
library(caret)
library(corrplot)
library(RColorBrewer)
library(cluster)
library(factoextra)
library(maps)
library(factoextra)
library(NbClust)
library(scales)
library(gridExtra)
library(grid)
library(lightgbm)
library(Matrix)
library(caret)
library(readxl)
library(lattice)
library(mapproj)

data <- read.csv("superstore_campaign_data.csv", fileEncoding="UTF-8-BOM")
salesdata <- read.csv("superstore_sales_data.csv")

head(data) 
str(data)
summary(data)
#handling missing data
sapply(data, function(x) sum(is.na(x)))
missing_data_percent <- round((sum(is.na(data$Income))/nrow(data))*100,2)
missing_data_percent
#we see there are 24 missing values in Income, which is equivalent to 1.1% of entire data frame
boxplot(data$Income,
        main = "Income distribution",
        col = "orange",
        border = "brown"
)
#we do see outliers in Income. Since missing data is only 1.1%, we can remove it.
data <- na.omit(data)
data2 <- data
#Add Age column (using current year - Year_Birth) & Customer Acct Age column (today - enrollment date)
data$Age <- 2023 - data$Year_Birth

data$Cust_Acct_Year <- format(as.Date(data$Dt_Customer, format="%m/%d/%Y"),"%Y")
table(data$Cust_Acct_Year)
data$Cust_Acct_Year <- as.numeric(data$Cust_Acct_Year)
data$Cust_Acct_Age <- 2023 - data$Cust_Acct_Year
#Interesting to see the income distribution between response groups
data$Response <- as.character(data$Response)
data$Response[data$Response == 1] <- 'Accept'
data$Response[data$Response == 0] <- 'Decline'
table(data$Response)
boxplot(data$Income ~ data$Response)
cor(data2$Income, data2$Response)
#confirmed there is no or weak correlation between Income and Response variables
#percentage of people who decline or accept the campaign?
response_table <- table(data$Response)
percentlabels<- round(100*response_table/sum(response_table), 1)
pie(response_table, labels = paste(c("Accept","Decline"),percentlabels, "%")) 
#only 15% accept the campaign

#Do people make complaint?
data$Complain <- as.character(data$Complain)
data$Complain[data$Complain == 1] <- 'Yes'
data$Complain[data$Complain == 0] <- 'No'
complain_table <- table(data$Complain)
complain_table
pie(complain_table, labels = paste(c("No","Yes"),round(100*complain_table/sum(complain_table), 1), "%")) 
#Interested to know response vs complain variables
ggplot(data, aes(Complain)) + geom_bar(aes(fill = Response), position = "dodge")
# --> people make no complaints tend to accept the campaign 

ggplot(data, aes(x = Recency, fill = Response)) +
  geom_histogram(position = "dodge", alpha = 1)
#Recency is number of days since the last purchase
#--> people who accept the campaign has smaller recency
ggplot(data, aes(x = Cust_Acct_Age, fill = Response)) +
  geom_histogram(position = "dodge", alpha = 1) + geom_bar(position = "dodge", width = 0.5)
#-> Customer with longer Customer Account Age tend to accept campaign

summary(data)
#Education, marital status, dt_customer, complain,   response



data<- data %>% 
  mutate(eductation_master_phd = ifelse(Education == 'Master'| Education == 'PhD', 1, 0)) %>% 
  mutate(education_graduation = ifelse(Education== 'Graduation', 1, 0))


table(data$Marital_Status)


data<- data %>% 
  mutate(marital_status_together = ifelse(Marital_Status == 'Married'| Marital_Status == 'Together', 1, 0)) %>% 
  mutate(marital_status_seperated = ifelse(Marital_Status== 'Divorced'| Marital_Status == 'Widow', 1, 0))


data<- data %>% 
  mutate(Complain = ifelse(Complain == 'Yes', 1, 0)) %>% 
  mutate(Response = ifelse(Response == 'Accept', 1, 0))

#$Complain <- data$Complain[data$Complain == 'Yes'] <- 1
#data$Complain <- data$Complain[data$Complain == 'No'] <- 0

#data$Response <- data$Complain[data$Complain == 'Accept'] <- 1
#data$Response <- data$Complain[data$Complain == 'Decline'] <- 0


data$Dt_Customer <- format(as.Date(data$Dt_Customer, format="%m/%d/%Y"))

latest_date <- max(data$Dt_Customer, na.rm = TRUE)

data$days_as_customer <- difftime( max(data$Dt_Customer, na.rm = TRUE), data$Dt_Customer, units = "days")

data$days_as_customer <- as.numeric(data$days_as_customer)
#Dropping columns we don't need anymore
drops <- c("Marital_Status","Education", "Dt_Customer")
data <- data[ , !(names(data) %in% drops)]

summary(data)

cor(data)

#GRANT
#using the "data" from Ben's "exploratory analysis.R
#split into training/test data
set.seed(123) 
train_indices <- sample(1:nrow(data), floor(0.7 * nrow(data)))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

library(ROSE)
table(train_data$Response)
desired_count <- 1551
#balance the dataset by oversampling Response = 1 and undersampling Response = 0
balanced_training_data <- ovun.sample(Response ~ ., data = train_data, method = "both", p = 0.5, N = desired_count)$data
table(balanced_training_data$Response)


#SVM model
library(kernlab)

svm_model_scaled <- ksvm(Response ~.,data = balanced_training_data,
                         type = "C-svc", 
                         kernel = "vanilladot", 
                         C = 100,
                         scaled=TRUE)

#calculate coefficients & intercept
a_scaled <- colSums(svm_model_scaled@xmatrix[[1]] * svm_model_scaled@coef[[1]])
a0_scaled<- -svm_model_scaled@b

#predict
pred_scaled <- predict(svm_model_scaled, balanced_training_data[,-18])
pred_scaled

#accuracy
sum(pred_scaled == balanced_training_data$Response) / nrow(balanced_training_data)
#accuracy = 0.80

#check different C values to be iterated over in the for loop
costs <- c(0.01, 1, 100, 1000, 10000, 100000, 1000000)
#empty list to hold the loop's output of the different accuracies for each C value
test_accuracy <- list()

for (i in 1:length(costs)) {
  model <- ksvm(as.matrix(balanced_training_data[,-18]), as.factor(balanced_training_data[,18]), type = "C-svc", kernel = "vanilladot", C = costs[i], scaled = TRUE)
  
  #see what the model predicts
  pred2 <- predict(model, balanced_training_data[,-18])
  
  # see what fraction of the model’s predictions match the actual classification
  acc <- sum(pred2 == balanced_training_data[,18]) / nrow(balanced_training_data)
  
  test_accuracy[[i]] <- acc
}

#see what the best accuracy is
test_accuracy

#best accuracy is close to C=1, which is what I started with. 

#working with prediction accuracy of 80% on the training data. 

#variable selection using forward-backward variable selection
library(dplyr)

predictor_cols <- setdiff(names(balanced_training_data), c("Id", "Response"))

# Create a new dataset with the standardized predictor variables
standardized_data <- balanced_training_data %>%
  mutate(across(all_of(predictor_cols), scale))

FitStart <- lm(Response~1, data = standardized_data)
FitAll <- lm(Response~., data = standardized_data)

step_model_1 <- stepAIC(FitStart, scope = formula(FitAll), direction = "both")
summary(step_model_1)

#run svm model with selected variables
svm_model_scaled2 <- ksvm(Response ~ days_as_customer + marital_status_together + NumWebVisitsMonth + NumStorePurchases + NumCatalogPurchases + MntGoldProds + MntFishProducts + MntMeatProducts + MntWines + Recency + Teenhome + Kidhome + Income, data = balanced_training_data,
                          type = "C-svc", 
                          kernel = "vanilladot", 
                          C = 1,
                          scaled=TRUE)

#predict
pred_scaled2 <- predict(svm_model_scaled2, balanced_training_data[,-18])
pred_scaled2

#accuracy
sum(pred_scaled2 == balanced_training_data$Response) / nrow(balanced_training_data)
#accuracy is 0.79. Slightly lower accuracy than the model with all predictors, but we will take the simpler model

#run model on test data
pred_test <- predict(svm_model_scaled2, test_data[, c(3,4,5,6,7,9,10,12,15,16,17,25,27)])

# Accuracy on the test data
accuracy_test <- sum(pred_test == test_data$Response) / nrow(test_data)
accuracy_test
#0.77. 

#cross-validate model with selected variables 
library(caret)

# Create the data matrix and response vector
X <- as.matrix(balanced_training_data[,c(3,4,5,6,7,9,10,12,15,16,17,25,27)])
y <- as.factor(balanced_training_data[,18])

num_folds <- 5  

# cross-validation object
cv <- trainControl(method = "cv", number = num_folds)

# train the SVM model using cross-validation
svm_model_cv <- train(x = X, y = y, method = "svmLinear", trControl = cv)

print(svm_model_cv)
#cv accuracy = 0.7814418


# Create the data matrix and response vector
data_matrix <- as.matrix(balanced_training_data[, c("days_as_customer", "marital_status_together", "NumWebVisitsMonth", "NumStorePurchases", "NumCatalogPurchases", "MntGoldProds", "MntFishProducts", "MntMeatProducts", "MntWines", "Recency", "Teenhome", "Kidhome", "Income")])
response_vector <- as.factor(balanced_training_data$Response)

# Make predictions using the SVM model
predictions <- predict(svm_model_scaled2, data_matrix)

# Create the confusion matrix
confusion_matrix <- confusionMatrix(factor(predictions, levels = levels(response_vector)), response_vector)

# Print the confusion matrix
print(confusion_matrix)
#Reference
#Prediction   0   1
#0 602 151
#1 173 625




#HAYDEN
# Decision Tree Model

library(rpart)
library(rpart.plot)
library(caret)

decision_tree <- rpart(Response ~., data = balanced_training_data, method = "class")
summary(decision_tree)
prp(decision_tree)
plotcp(decision_tree)

# Pruning the Tree

pruned_tree <- prune(decision_tree, cp = 0.015)
prp(pruned_tree)

# Testing the model

predicted_classes <- predict(pruned_tree, test_data, type = "class")
predicted_classes
actual_classes <- as.factor(test_data$Response)
actual_classes

confusion_matrix_ct <- confusionMatrix(predicted_classes,actual_classes)
confusion_matrix_ct

#Confusion Matrix and Statistics

#Reference
#Prediction   0   1
#          0 402  27
#         1 175  61

#Accuracy : 0.6962         
#95% CI : (0.6597, 0.731)
#No Information Rate : 0.8677         
#P-Value [Acc > NIR] : 1              

#Kappa : 0.2277         

#Mcnemar's Test P-Value : <2e-16         

#Sensitivity : 0.6967         
#Specificity : 0.6932         


#NGOC  
#logistic regression model
fit.full <- glm(Response ~.,family=binomial(link='logit'),data=balanced_training_data)

summary(fit.full)

fit.reduced <- glm(Response ~ Teenhome+
                     Recency+MntWines+MntMeatProducts+MntSweetProducts+
                     MntGoldProds+NumCatalogPurchases+NumStorePurchases+NumDealsPurchases+
                     NumWebVisitsMonth+days_as_customer+marital_status_together+eductation_master_phd+Income+MntFishProducts+Kidhome ,family=binomial(link='logit'),data=balanced_training_data)

summary(fit.reduced)


#After removing multiple variables that are not significant, we finally got better fit model
anova(fit.reduced, fit.full, test="Chisq") 
#The output above displays nonsignificant chi-square value with p-values= 0.62. 
#It means that the second model with only significant predictors fits as well as the full model. 
#It supports our initial belief that Year_Birth,Income, Kidhome, MntFruits, MntFishProducts, NumWebPurchases, Complain don’t add any contribution to predict infidelity (our response variable). 
#Thus, we will continue the analysis with the fit.reduced model as it is easier to do our interpretations on the simpler model.
#testing accurancy of training data
balanced_training_data$prob <- predict(fit.reduced, newdata=balanced_training_data,
                                       type="response")

train <- balanced_training_data  %>% mutate(model_pred = 1*(prob > .50) + 0)

train <- train %>% mutate(accurate = 1*(model_pred == Response))
(sum(train$accurate)/nrow(train))

test_data$prob <- predict(fit.reduced, newdata=test_data,
                          type="response")

test <- test_data  %>% mutate(model_pred = 1*(prob > .50) + 0)

test <- test %>% mutate(accurate = 1*(model_pred == Response))
(sum(test$accurate)/nrow(test))

#test$Response <- ifelse(test$Response=="Yes", 1, 0)

#optimal <- optimalCutoff(test$Response, test_data$model_pred)[1]


#HOA
#Light Gradient Boosting Model

# Install needed packages
library(dplyr)
# Check duplicated values
sum(duplicated(data2))
# Check the values in marital status column
unique(data2$Marital_Status)

#Removing values false values in marital status column, combines value "Together" and "Married"
datam<-subset(data2, Marital_Status!="YOLO" & Marital_Status!="Alone" & Marital_Status!="Absurd")
datam$Marital_Status<-str_replace(datam$Marital_Status, "Together","Married")
unique(datam$Marital_Status)


# Distribution of each products
ggplot(data, aes(x = MntWines)) +
  geom_density(fill = "blue") +
  labs(title = "Wine Products Distribution", x = "Amount Spent")
ggplot(data, aes(x = MntFruits)) +
  geom_density(fill = "blue") +
  labs(title = "Fruit Products Distribution", x = "Amount Spent")
ggplot(data, aes(x = MntFishProducts)) +
  geom_density(fill = "blue") +
  labs(title = "Fish Products Distribution", x = "Amount Spent")
ggplot(data, aes(x = MntMeatProducts)) +
  geom_density(fill = "blue") +
  labs(title = "Meat Products Distribution", x = "Amount Spent")
ggplot(data, aes(x = MntSweetProducts)) +
  geom_density(fill = "blue") +
  labs(title = "Sweet Products Distribution", x = "Amount Spent")
ggplot(data, aes(x = MntGoldProds)) +
  geom_density(fill = "blue") +
  labs(title = "Gold Products Distribution", x = "Amount Spent")


df = data2 %>% 
  mutate(Response = factor(Response, labels = c("No Response", "Response")),
         Education = factor(Education))
ggplot(df, aes(x = Education, fill = Response)) +
  geom_bar(position = position_dodge()) +
  theme_classic()

df = data2 %>% 
  mutate(Response = factor(Response, labels = c("No Response", "Response")),
         Marital_Status = factor(Marital_Status))
ggplot(df, aes(x = Marital_Status, fill = Response)) +
  geom_bar(position = position_dodge()) +
  theme_classic()

# Check the data type
str(datam)

#Removing Id and Dt_Customer Columns
data_remove <- datam[, -which(names(datam) == "Id")]
data_remove <- data_remove[, -which(names(data_remove) == "Dt_Customer")]


str(data_remove)
# Change categorical variables from chr type to numeric
data_remove$Marital_Status <- as.numeric(factor(data_remove$Marital_Status))
data_remove$Education <- as.numeric(factor(data_remove$Education))

data_remove_cor <- cor(data_remove)
corrplot(data_remove_cor)
#From the output, we have negative correlation for some pairs of variables such as Income and the number of monthly web visit,number of monthly web visit and products of MntWine; MntFrui; MntMeat; MntSweet.
#We have positive correlation for some pairs of variables such as Income and products of MntWine; MntMeat, income and number of catalog purchase, income and number of store purchase.

# Build Light Gradient Boosting Model
library(ROSE)
library(lightgbm)
library(Matrix)
library(caret)
library(readxl)
library(tidyverse)
library(ggplot2)
library(lattice)
library(caret)
library(dplyr)

# Build the model before balancing dataset
set.seed(123) 
train_indices <- sample(1:nrow(data_remove), floor(0.7 * nrow(data_remove)))
train_data <- data_remove[train_indices, ]
test_data <- data_remove[-train_indices, ]
train = sparse.model.matrix(Response ~., data = train_data)
train_y = train_data[,"Response"]
test = sparse.model.matrix(Response~., data= test_data)
test_y = test_data[,"Response"]
train_matrix = lgb.Dataset(data = as.matrix(train), label = train_y)
test_matrix = lgb.Dataset(data = as.matrix(test), label = test_y)
t = list(test = test_matrix)
params = list(max_bin = 10,
              learning_rate = 0.001,
              objective = "binary",
              metric = 'binary_logloss')
bst = lightgbm(params = params, train_matrix, t , nrounds = 1000)
p = predict(bst, test)
test_data$predicted = ifelse(p > 0.5,1,0)
cmlgb = confusionMatrix(factor(test_data$predicted), factor(test_data$Response))
cmlgb
# Extract metrics
cmlgb$byClass['Recall']
cmlgb$byClass['Precision']
cmlgb$byClass['F1']

#visualize the confusion matrix 
df_cmlgb <- as.data.frame(cmlgb$table)
heatmap_cmlgb <- ggplot(df_cmlgb, aes(x = Prediction, y = Reference)) +
  geom_tile(aes(fill = log(Freq + 1)), color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 4) +
  scale_fill_gradient(low = "white", high = "pink") +
  labs(title = "LightGBM Confusion Matrix before balancing dataset", x = "Predicted", y = "True")
print(heatmap_cmlgb)

# Build LGB Model after balancing the dataset using ROSE package
balanced_data <- ROSE(Response ~ ., data = train_data)$data
table(balanced_data$Response)
trainb <- sparse.model.matrix(Response ~ ., data = balanced_data)
trainb_y <- balanced_data[,"Response"]
testb <- sparse.model.matrix(Response ~ ., data = test_data)
testb_y <- test_data[,"Response"]
trainb_matrix <- lgb.Dataset(data = as.matrix(trainb), label = trainb_y)
testb_matrix <- lgb.Dataset(data = as.matrix(testb), label = testb_y)
tb <- list(test = testb_matrix)
params <- list(max_bin = 10,
               learning_rate = 0.001,
               objective = "binary",
               metric = 'binary_logloss')
bst <- lightgbm(params = params, trainb_matrix, tb, nrounds = 1000)
# Remove the "predicted" column from the validation data
testb <- testb[, -which(colnames(testb) == "predicted")]
p <- predict(bst, testb)
test_data$predicted <- ifelse(p > 0.5, 1, 0)
cmlgb_balanced <- confusionMatrix(factor(test_data$predicted), factor(test_data$Response))
cmlgb_balanced
# Extract metrics
cmlgb_balanced$byClass['Recall']
cmlgb_balanced$byClass['Precision']
cmlgb_balanced$byClass['F1']

# Visialize the confusion matrix for data after balancing
df_cmlgb_balanced <- as.data.frame(cmlgb_balanced$table)
heatmap_cmlgb_balanced <- ggplot(df_cmlgb_balanced, aes(x = Prediction, y = Reference)) +
  geom_tile(aes(fill = log(Freq + 1)), color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 4) +
  scale_fill_gradient(low = "white", high = "pink") +
  labs(title = "LightGBM Confusion Matrix after balancing dataset", x = "Predicted", y = "True")
print(heatmap_cmlgb_balanced)

library(ggplot2)
# Create a data frame with the metrics
metrics <- data.frame(
  Model = c("Before Balancing", "After Balancing"),
  Accuracy = c(0.8688, 0.7888),
  Precision = c(0.8712, 0.9336),
  Recall = c(0.9947, 0.8126),
  F1_Score = c(0.9289, 0.8689)
)
# Reshape the data frame into a long format
metrics_long <- tidyr::gather(metrics, Metric, Value, -Model)
# Create a bar plot using ggplot2
plot <- ggplot(metrics_long, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Metrics", y = "Metric Values", title = "Comparison of LGB Model Performance") +
  scale_fill_manual(values = c("lightblue", "blue")) +
  theme_minimal()
# Display the plot
print(plot)

# Metrics for SVM
svm_sensitivity <- 0.7767742
svm_precision <- 0.7994688
svm_f1 <- 0.7879581

# Metrics for Decision Tree
dt_sensitivity <- 0.6967071
dt_precision <- 0.9370629
dt_f1 <- 0.7992048

# Metrics for Logistic Regression
lr_sensitivity <- 0.7694974
lr_precision <- 0.9568966
lr_f1 <- 0.8530259

# Metrics for Light Gradient Boosting
lgb_precision <- 0.862068965517241
lgb_recall <- 0.956284153005464
lgb_f1 <- 0.906735751295337

# Labels for the models
models <- c('SVM', 'Decision Tree', 'Logistic Regression', 'LightGBM')

# Data for each metric
sensitivity_values <- c(svm_sensitivity, dt_sensitivity, lr_sensitivity, lgb_recall)
precision_values <- c(svm_precision, dt_precision, lr_precision, lgb_precision)
f1_values <- c(svm_f1, dt_f1, lr_f1, lgb_f1)

# Load necessary library
library(ggplot2)

# Function to create a bar chart for a specific metric
plot_metric <- function(metric_values, metric_name) {
  data <- data.frame(models, metric_values)
  ggplot(data, aes(x = models, y = metric_values * 100, fill = models)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = paste0(round(metric_values * 100, 1), "%")), vjust = -0.5) +
    labs(x = "Models", y = paste(metric_name, "(%)"), title = paste("Model Comparison", metric_name, "for Different Models")) +
    theme_minimal()
}

# Create bar charts for each metric
plot_metric(sensitivity_values, "Sensitivity")
plot_metric(precision_values, "Precision")
plot_metric(f1_values, "F1-Score")



# Metrics for SVM
svm_sensitivity <- 0.7767742
svm_precision <- 0.7994688
svm_f1 <- 0.7879581

# Metrics for Decision Tree
dt_sensitivity <- 0.6967071
dt_precision <- 0.9370629
dt_f1 <- 0.7992048

# Metrics for Logistic Regression
lr_sensitivity <- 0.7694974
lr_precision <- 0.9568966
lr_f1 <- 0.8530259

# Metrics for Light Gradient Boosting
lgb_precision <- 0.862068965517241
lgb_recall <- 0.956284153005464
lgb_f1 <- 0.906735751295337

# Labels for the models
models <- c('SVM', 'Decision Tree', 'Logistic Regression', 'LightGBM')

# Create a data frame to compare metrics for different models
metrics_table <- data.frame(
  Model = models,
  Sensitivity = c(svm_sensitivity, dt_sensitivity, lr_sensitivity, lgb_recall),
  Precision = c(svm_precision, dt_precision, lr_precision, lgb_precision),
  F1_Score = c(svm_f1, dt_f1, lr_f1, lgb_f1)
)

# Print the metrics table
print(metrics_table)


#BEN
salesdata <- read.csv("superstore_sales_data.csv")

us_state_to_abbrev <- c(
  "Alabama" = "AL",
  "Alaska" = "AK",
  "Arizona" = "AZ",
  "Arkansas" = "AR",
  "California" = "CA",
  "Colorado" = "CO",
  "Connecticut" = "CT",
  "Delaware" = "DE",
  "Florida" = "FL",
  "Georgia" = "GA",
  "Hawaii" = "HI",
  "Idaho" = "ID",
  "Illinois" = "IL",
  "Indiana" = "IN",
  "Iowa" = "IA",
  "Kansas" = "KS",
  "Kentucky" = "KY",
  "Louisiana" = "LA",
  "Maine" = "ME",
  "Maryland" = "MD",
  "Massachusetts" = "MA",
  "Michigan" = "MI",
  "Minnesota" = "MN",
  "Mississippi" = "MS",
  "Missouri" = "MO",
  "Montana" = "MT",
  "Nebraska" = "NE",
  "Nevada" = "NV",
  "New Hampshire" = "NH",
  "New Jersey" = "NJ",
  "New Mexico" = "NM",
  "New York" = "NY",
  "North Carolina" = "NC",
  "North Dakota" = "ND",
  "Ohio" = "OH",
  "Oklahoma" = "OK",
  "Oregon" = "OR",
  "Pennsylvania" = "PA",
  "Rhode Island" = "RI",
  "South Carolina" = "SC",
  "South Dakota" = "SD",
  "Tennessee" = "TN",
  "Texas" = "TX",
  "Utah" = "UT",
  "Vermont" = "VT",
  "Virginia" = "VA",
  "Washington" = "WA",
  "West Virginia" = "WV",
  "Wisconsin" = "WI",
  "Wyoming" = "WY",
  "District of Columbia" = "DC",
  "American Samoa" = "AS",
  "Guam" = "GU",
  "Northern Mariana Islands" = "MP",
  "Puerto Rico" = "PR",
  "United States Minor Outlying Islands" = "UM",
  "U.S. Virgin Islands" = "VI"
)

# Create a new column with State Abbreviations
salesdata$State_abb <- us_state_to_abbrev[salesdata$State]

# Convert date variables to date format
salesdata$Order.Date <- format(as.Date(salesdata$Order.Date, format="%m/%d/%Y"))
salesdata$Ship.Date <- format(as.Date(salesdata$Ship.Date, format="%m/%d/%Y"))
salesdata$discount_dollars <- salesdata$Sales * salesdata$Discount
head(salesdata) 
str(salesdata)
summary(salesdata)




# Lowecase columns and replace spaces with underscores
colnames(salesdata) <- tolower(colnames(salesdata))
colnames(salesdata) <- gsub(" ", "_", colnames(salesdata))

head(salesdata)
#Add new column and name it "profit.ratio"
salesdata$profit.ratio <- salesdata$profit/salesdata$sales
# Check the new data
head(salesdata)

# Check for missing values
sum(is.na(salesdata))





########################### Understanding the Shape of the data ################


#Distribution of Sales
ggplot(salesdata, aes(x = sales)) +
  geom_histogram(fill = "steelblue", color = "white", bins = 30) +
  labs(title = "Distribution of Sales", x = "Sales", y = "Frequency") +
  theme_minimal() +
  scale_x_continuous(limits = c(0, 2500)) +
  scale_fill_continuous(limits = c(0, 2500))



#Distribution of Profit
ggplot(salesdata, aes(x = profit)) +
  geom_histogram(fill = "steelblue", color = "white") +
  labs(title = "Distribution of Profit", x = "Profit", y = "Frequency") +
  theme_minimal() +
  scale_x_continuous(limits = c(-400, 400)) +
  scale_fill_continuous(limits = c(-400, 00))


#Frequency by Ship Mode
ggplot(salesdata, aes(x = ship.mode)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Frequency of Orders by Ship Mode", x = "Ship Mode", y = "Frequency") +
  theme_minimal()

# The Standard Class was the most popular shipping.

#Frequency by Segment
ggplot(salesdata, aes(x = segment)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Frequency of Orders by Segment", x = "Segment", y = "Frequency") +
  theme_minimal()

#Consumer is the most popular


#Frequency by Region
ggplot(salesdata, aes(x = region)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Frequency of Orders by Region", x = "Region", y = "Frequency") +
  theme_minimal()

#West is the most popular

#Frequency by state
state_counts <- table(salesdata$`state`)

# Order states based on frequency 
state_ordered <- names(state_counts)[order(state_counts, decreasing = TRUE)]

# Convert Sub Category to a factor with ordered levels
salesdata$`state` <- factor(salesdata$`state`, levels = state_ordered)

# Create the bar chart with ordered bars
ggplot(salesdata, aes(x = `state`, fill = `state`)) +
  geom_bar() +
  labs(title = "Frequency of Orders by State", x = "State", y = "Frequency") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  guides(fill = FALSE)




#Frequency by Category
ggplot(salesdata, aes(x = category)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Frequency of Orders by Category", x = "Category", y = "Frequency") +
  theme_minimal()



#Frequency by sub Category
subcategory_counts <- table(salesdata$`sub.category`)

# Order Sub Category based on frequency 
subcategory_ordered <- names(subcategory_counts)[order(subcategory_counts, decreasing = TRUE)]

# Convert Sub Category to a factor with ordered levels
salesdata$`sub.category` <- factor(salesdata$`sub.category`, levels = subcategory_ordered)

# Create the bar chart with ordered bars
ggplot(salesdata, aes(x = `sub.category`, fill = `sub.category`)) +
  geom_bar() +
  labs(title = "Frequency of Orders by Sub Category", x = "Sub Category", y = "Frequency") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  guides(fill = FALSE)

#Binders are the most popular sub category

#Inference:
#Three most popular states among the customers are California, New York and Texas.
#Majority of the customers prefer the Standard Class Shipping mode.
#Most of the customers are from Consumer Segment.
#Least of all orders from the southern region.
#Office Supplies take up most of the sales.
#Blinders and Paper are clear leaders in sales among customers.


################## GEO ANALYSIS ##################################

##### Region Analysis

region_sales <- salesdata %>%
  group_by(region) %>%
  summarise(Sales = sum(sales),
            Profit = sum(profit),
            Profit_Ratio = sum(profit)/sum(sales))


# Order the data by Sales in descending order
region_sales_sales_order <- region_sales[order(-region_sales$Sales), ]
# Order the data by Profit in descending order
region_sales_profit_order <- region_sales[order(-region_sales$Profit), ]
# Order the data by Profit Ratio in descending order
region_sales_ratio_order <- region_sales[order(-region_sales$Profit_Ratio), ]

# Bar chart for Sales by region
ggplot(region_sales_sales_order, aes(x = reorder(region, -Sales), y = Sales)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Sales by Region", x = "Region", y = "Sales") +
  theme_minimal() +
  coord_flip() +
  scale_y_continuous(labels = scales::comma)

# Bar chart for Profit by region
ggplot(region_sales_profit_order, aes(x = reorder(region, -Profit), y = Profit)) +
  geom_bar(stat = "identity", fill = "green") +
  labs(title = "Total Profit by Region", x = "Region", y = "Profit") +
  theme_minimal() +
  coord_flip() +
  scale_y_continuous(labels = scales::comma)

# Bar chart for Profit Ratio by region
ggplot(region_sales_ratio_order, aes(x = reorder(region, -Profit_Ratio), y = Profit_Ratio)) +
  geom_bar(stat = "identity", fill = "orange") +
  labs(title = "Profit Ratio by Region", x = "Region", y = "Profit Ratio") +
  theme_minimal() +
  coord_flip()



##### State Analysis


state_sales <- salesdata %>%
  group_by(state) %>%
  summarise(Sales = sum(sales),
            Profit = sum(profit),
            Profit_Ratio = sum(profit)/sum(sales))

state_sales$state <- tolower(state_sales$state)
map_data <- map_data("state")
state_sales <- merge(map_data, state_sales, by.x = "region", by.y = "state", all.x = TRUE)

# Heat Map of Sales
ggplot(state_sales, aes(x = long, y = lat, group = group, fill = Sales)) +
  geom_polygon(color = "white", size = 0.5) +  # Add state outlines with white color and increased size
  labs(title = "Total Sales by State", fill = "Sales") +
  scale_fill_gradient(low = "lightblue", high = "darkblue", labels = scales::comma) +  # Format labels with comma separator
  theme_void() +
  coord_map()


# Heat Map of Profit
ggplot(state_sales, aes(x = long, y = lat, group = group, fill = Profit)) +
  geom_polygon(color = "white", size = 0.5) +  # Add state outlines with white color and increased size
  labs(title = "Total Profit by State", fill = "Profit") +
  scale_fill_gradient(low = "lightblue", high = "darkblue", labels = scales::comma) +  # Format labels with comma separator
  theme_void() +
  coord_map()




# Heat Map of Profit Ratio
ggplot(state_sales, aes(x = long, y = lat, group = group, fill = Profit_Ratio)) +
  geom_polygon(color = "white", size = 0.5) +
  labs(title = "Total Profitability by State", fill = "Profit Ratio") +
  scale_fill_gradient2(low = "darkred", mid = "grey", high = "darkgreen", midpoint = 0,
                       limits = c(min(state_sales$Profit_Ratio), max(state_sales$Profit_Ratio)),
                       labels = scales::percent_format(accuracy = 0.01)) +
  theme_void() +
  coord_map()


# There are states that are not profitable


################## PRODUCT ANALYSIS ##################################

##### Category Analysis

category_sales <- salesdata %>%
  group_by(category) %>%
  summarise(Sales = sum(sales),
            Profit = sum(profit),
            Profit_Ratio = sum(profit)/sum(sales))


# Order the data by Sales in descending order
category_sales_sales_order <- category_sales[order(-category_sales$Sales), ]
# Order the data by Profit in descending order
category_sales_profit_order <- category_sales[order(-category_sales$Profit), ]
# Order the data by Profit Ratio in descending order
category_sales_ratio_order <- category_sales[order(-category_sales$Profit_Ratio), ]

# Bar chart for Sales by category
ggplot(category_sales_sales_order, aes(x = reorder(category, -Sales), y = Sales)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Sales by Category", x = "Category", y = "Sales") +
  theme_minimal() +
  coord_flip() +
  scale_y_continuous(labels = scales::comma) 

# Bar chart for Profit by category
ggplot(category_sales_profit_order, aes(x = reorder(category, -Profit), y = Profit)) +
  geom_bar(stat = "identity", fill = "green") +
  labs(title = "Total Profit by Category", x = "Category", y = "Profit") +
  theme_minimal() +
  coord_flip() +
  scale_y_continuous(labels = scales::comma)

# Bar chart for Profit Ratio by category
ggplot(category_sales_ratio_order, aes(x = reorder(category, -Profit_Ratio), y = Profit_Ratio)) +
  geom_bar(stat = "identity", fill = "orange") +
  labs(title = "Profit Ratio by Category", x = "Category", y = "Profit Ratio") +
  theme_minimal() +
  coord_flip() +
  scale_y_continuous(labels = scales::percent_format(accuracy = 0.01))


#Furniture is a low profit margin

##### Sub Category Analysis


sub_category_sales <- salesdata %>%
  group_by(sub.category) %>%
  summarise(Sales = sum(sales),
            Profit = sum(profit),
            Discount_Dollars = sum(discount_dollars),
            Discount_Ratio = sum(discount_dollars)/sum(sales),
            Profit_Ratio = sum(profit)/sum(sales))



# Order the data by Sales in descending order
sub_category_sales_sales_order <- sub_category_sales[order(-sub_category_sales$Sales), ]
# Order the data by Profit in descending order
sub_category_sales_profit_order <- sub_category_sales[order(-sub_category_sales$Profit), ]
# Order the data by Profit Ratio in descending order
sub_category_sales_ratio_order <- sub_category_sales[order(-sub_category_sales$Profit_Ratio), ]
# Order the data by Discount Ratio in descending order
sub_category_sales_discount_order <- sub_category_sales[order(-sub_category_sales$Discount_Ratio), ]

# Bar chart for Sales by category
ggplot(sub_category_sales_sales_order, aes(x = reorder(sub.category, -Sales), y = Sales)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Sales by Sub Category", x = "Sub Category", y = "Sales") +
  theme_minimal() +
  coord_flip() +
  scale_y_continuous(labels = scales::comma) 

# Bar chart for Profit by  sub category
ggplot(sub_category_sales_profit_order, aes(x = reorder(sub.category, -Profit), y = Profit)) +
  geom_bar(stat = "identity", fill = "green") +
  labs(title = "Total Profit by Sub Category", x = "Sub Category", y = "Profit") +
  theme_minimal() +
  coord_flip() +
  scale_y_continuous(labels = scales::comma)

# Bar chart for Profit Ratio by sub category
ggplot(sub_category_sales_ratio_order, aes(x = reorder(sub.category, -Profit_Ratio), y = Profit_Ratio)) +
  geom_bar(stat = "identity", fill = "orange") +
  labs(title = "Profit Ratio by Sub Category", x = "Sub Category", y = "Profit Ratio") +
  theme_minimal() +
  coord_flip() +
  scale_y_continuous(labels = scales::percent_format(accuracy = 0.01))


# Bar chart for Discount Ratio by sub category
ggplot(sub_category_sales_discount_order, aes(x = reorder(sub.category, -Discount_Ratio), y = Discount_Ratio)) +
  geom_bar(stat = "identity", fill = "orange") +
  labs(title = "Discount Ratio by Sub Category", x = "Sub Category", y = "Profit Ratio") +
  theme_minimal() +
  coord_flip() +
  scale_y_continuous(labels = scales::percent_format(accuracy = 0.01))


# Select the columns for profit ratio and discount ratio
sub_category_sales_ratio_order <- sub_category_sales_ratio_order[, c("sub.category", "Profit_Ratio")]
sub_category_sales_discount_order <- sub_category_sales_discount_order[, c("sub.category", "Discount_Ratio")]

# Merge the data frames by sub-category
cor_data <- merge(sub_category_sales_ratio_order, sub_category_sales_discount_order, by = "sub.category")

# Compute the correlation matrix
cor_matrix <- cor(cor_data[, c("Profit_Ratio", "Discount_Ratio")])

# Display the correlation matrix
cor_matrix

#Furniture sub categories not profitable

##### Product Analysis

product_sales <- salesdata %>%
  group_by(product.name) %>%
  summarise(Sales = sum(sales),
            Profit = sum(profit),
            Profit_Ratio = sum(profit)/sum(sales))



#top 50 in sales
top_sales_products <- product_sales %>%
  arrange(desc(Sales)) %>%
  head(75)

#worst 15 sales in profit ratio from the top 75 best selling products
worst_products <- top_sales_products %>%
  arrange(Profit_Ratio) %>%
  head(15)

#all have a negative profit ratio, losing products
ggplot(worst_products, aes(x = reorder(product.name, Profit_Ratio), y = Profit_Ratio)) +
  geom_bar(stat = "identity", fill = "red") +
  labs(title = "Worst Products by Profit Ratio with Sales listed", x = "Product", y = "Profit Ratio") +
  theme_minimal() +
  coord_flip() +
  scale_y_continuous(labels = percent_format(accuracy = 0.01)) +
  geom_text(aes(label = scales::comma(Sales), y = Profit_Ratio), 
            hjust = 0, vjust = 0.5, color = "black", size = 3, fontface = "bold")


## 15 most unprofitable products

################## CLUSTERING ##################################

##### 1 Data Pre-Processing

sales <- read.csv("superstore_sales_data.csv")

sales$Order.Date <- format(as.Date(sales$Order.Date, format="%m/%d/%Y"))
sales$Ship.Date <- format(as.Date(sales$Ship.Date, format="%m/%d/%Y"))

latest_date <- max(sales$Ship.Date, na.rm = TRUE)

print(latest_date)

aggregated_data <- sales %>%
  group_by(Customer.ID, Customer.Name,Segment) %>%
  summarise(
    total_sales = sum(Sales),
    Total_profit = sum(Profit),
    Total_Quantity = sum(Quantity),
    Total_Orders = n_distinct(Order.ID),
    Total_Sub_Categories = n_distinct(Sub.Category),
    Total_Unique_Products = n_distinct(Product.Name),
    Furniture_Sales = sum(ifelse(Category == 'Furniture', Sales, 0)),
    Office_Sales = sum(ifelse(Category == 'Office Supplies', Sales, 0)),
    Technology_Sales = sum(ifelse(Category == 'Technology', Sales, 0)),
    Furniture_Profit = sum(ifelse(Category == 'Furniture', Profit, 0)),
    Office_Profit = sum(ifelse(Category == 'Office Supplies', Profit, 0)),
    Technology_Proft = sum(ifelse(Category == 'Technology', Profit, 0)),
    Furniture_Quantity = sum(ifelse(Category == 'Furniture', Quantity, 0)),
    Office_Quantity = sum(ifelse(Category == 'Office Supplies', Quantity, 0)),
    Technology_Quantity = sum(ifelse(Category == 'Technology', Quantity, 0)),
    Furniture_Orders = n_distinct(ifelse(Category == 'Furniture', Order.ID, 0)),
    Office_Orders = n_distinct(ifelse(Category == 'Office Supplies',Order.ID, 0)),
    Technology_Orders = n_distinct(ifelse(Category == 'Technology', Order.ID, 0)),
    
    Central_Sales = sum(ifelse(Region == 'Central', Sales, 0)),
    East_Sales = sum(ifelse(Region == 'East', Sales, 0)),
    South_Sales = sum(ifelse(Region == 'South', Sales, 0)),
    West_Sales = sum(ifelse(Region == 'West', Sales, 0)),
    Central_Profit = sum(ifelse(Region == 'Central', Profit, 0)),
    East_Profit = sum(ifelse(Region == 'East', Profit, 0)),
    South_Profit = sum(ifelse(Region == 'South', Profit, 0)),
    West_Profit = sum(ifelse(Region == 'West', Profit, 0)),
    Central_Quantity = sum(ifelse(Region == 'Central', Quantity, 0)),
    East_Quantity = sum(ifelse(Region == 'East', Quantity, 0)),
    South_Quantity = sum(ifelse(Region == 'South', Quantity, 0)),
    West_Quantity = sum(ifelse(Region == 'West', Quantity, 0)),
    Central_Orders = n_distinct(ifelse(Region == 'Central', Order.ID, 0)),
    East_Orders = n_distinct(ifelse(Region == 'East', Order.ID, 0)),
    South_Orders = n_distinct(ifelse(Region == 'South',Order.ID, 0)),
    West_Orders = n_distinct(ifelse(Region == 'West', Order.ID, 0)),
    recency = as.integer(round(difftime(latest_date,max(Order.Date, na.rm = TRUE), units = "days"))),
    days_as_customer = as.integer(round(difftime(latest_date,min(Order.Date, na.rm = TRUE), units = "days")))
    
  )

aggregated_data<- aggregated_data%>% 
  mutate(Segment_Consumer = ifelse(Segment == 'Consumer', 1, 0)) %>% 
  mutate(Segment_Corporate = ifelse(Segment == 'Corporate', 1, 0))

#Dont need segment anymore or customer name. Keeping customer id for identification later but we won't cluster on it
drops <- c("Segment","Customer.Name")
aggregated_data <- aggregated_data[ , !(names(aggregated_data) %in% drops)]



# Lowecase columns and replace spaces with underscores
colnames(aggregated_data) <- tolower(colnames(aggregated_data))
colnames(aggregated_data) <- gsub(" ", "_", colnames(aggregated_data))


#793 Customers to cluster
summary(aggregated_data)

##### 2 Variable Selection

#Will try an RFM approach, an all feature approach, and a PCA approach
RFM <- aggregated_data[, c("recency", "total_orders", "total_profit")]
cluster_data <- aggregated_data[, -1]

####### Run PCA
cluster_data_pca <- cluster_data
pca <- prcomp(cluster_data, scale = TRUE)
summary(pca)

variance_explained <- pca$sdev^2

screeplot(pca, type="lines",col="purple")
cumulative_variance <- cumsum(variance_explained)

# Create a dataframe for plotting
df <- data.frame(Component = 1:length(variance_explained),
                 VarianceExplained = cumulative_variance)

# Plot the explained variance by component
ggplot(df, aes(x = Component, y = VarianceExplained)) +
  geom_line() +
  geom_point() +
  labs(x = "Component", y = "Cumulative Variance Explained",
       title = "Explained Variance by Component") +
  theme_minimal()


pca_data <- data.frame(pca$x[, 1:12]) #taking first 12 pc's which explain 80% of variance


#putting all data on the same scale
scaled_rfm <- scale(RFM)
scaled_data <- scale(cluster_data)

#RFM[, c("recency", "total_orders", "total_profit")] <- scaled_rfm
#cluster_data[, ,] <- scaled_data

#PCA already scaled


##### 3 Finding optimal number of clusters 

### All Features
iss <- function(k) {
  kmeans(scaled_data,k,iter.max=100,nstart=100,algorithm ='Lloyd')$tot.withinss
}

k.values <- 1:10

iss_values <- map_dbl(k.values,iss)

plot(k.values,iss_values,
     type ='b', pch = 19, frame = FALSE,
     xlab= "Number of clusters K",
     ylab = 'Total intra-clusters sum of squares')

#average silhouette method



k2 <- kmeans(scaled_data,2,iter.max=100,nstart=50,algorithm ='Lloyd')
s2 <- plot(silhouette(k2$cluster,dist(scaled_data,'euclidean')))

k3 <- kmeans(scaled_data,3,iter.max=100,nstart=50,algorithm ='Lloyd')
s3 <- plot(silhouette(k3$cluster,dist(scaled_data,'euclidean')))

k4 <- kmeans(scaled_data,4,iter.max=100,nstart=50,algorithm ='Lloyd')
s4 <- plot(silhouette(k4$cluster,dist(scaled_data,'euclidean')))

k5 <- kmeans(scaled_data,5,iter.max=100,nstart=50,algorithm ='Lloyd')
s5 <- plot(silhouette(k5$cluster,dist(scaled_data,'euclidean')))

k6 <- kmeans(scaled_data,6,iter.max=100,nstart=50,algorithm ='Lloyd')
s6 <- plot(silhouette(k6$cluster,dist(scaled_data,'euclidean')))

k7 <- kmeans(scaled_data,7,iter.max=100,nstart=50,algorithm ='Lloyd')
s7 <- plot(silhouette(k7$cluster,dist(scaled_data,'euclidean')))

k8 <- kmeans(scaled_data,8,iter.max=100,nstart=50,algorithm ='Lloyd')
s8 <- plot(silhouette(k8$cluster,dist(scaled_data,'euclidean')))

k9 <- kmeans(scaled_data,9,iter.max=100,nstart=50,algorithm ='Lloyd')
s9 <- plot(silhouette(k9$cluster,dist(scaled_data,'euclidean')))

k10 <- kmeans(scaled_data,10,iter.max=100,nstart=50,algorithm ='Lloyd')
s10 <- plot(silhouette(k10$cluster,dist(scaled_data,'euclidean')))



fviz_nbclust(scaled_data,kmeans,method = 'silhouette')

nb_results <- NbClust(scaled_data, diss = NULL, distance = 'euclidean', min.nc = 2, max.nc = 10,
                      method = 'kmeans', index = 'silhouette')

fviz_nbclust(nb_results, geom = "point", show.clustvar = FALSE)

#gap statistic method
set.seed(125)
stat_gap <- clusGap(scaled_data, FUN = kmeans, nstart = 25, K.max = 10)

fviz_gap_stat(stat_gap)

#Optimal is 2


### RFM Method
iss <- function(k) {
  kmeans(scaled_rfm,k,iter.max=100,nstart=100,algorithm ='Lloyd')$tot.withinss
}

k.values <- 1:10

iss_values <- map_dbl(k.values,iss)

plot(k.values,iss_values,
     type ='b', pch = 19, frame = FALSE,
     xlab= "Number of clusters K",
     ylab = 'Total intra-clusters sum of squares')

#average silhouette method

library(cluster)
library(gridExtra)
library(grid)

k2 <- kmeans(scaled_rfm,2,iter.max=100,nstart=50,algorithm ='Lloyd')
s2 <- plot(silhouette(k2$cluster,dist(scaled_rfm,'euclidean')))

k3 <- kmeans(scaled_rfm, 3, iter.max = 100, nstart = 50, algorithm = 'Lloyd')
s3 <- plot(silhouette(k3$cluster, dist(scaled_rfm, 'euclidean')))

k4 <- kmeans(scaled_rfm, 4, iter.max = 100, nstart = 50, algorithm = 'Lloyd')
s4 <- plot(silhouette(k4$cluster, dist(scaled_rfm, 'euclidean')))

k5 <- kmeans(scaled_rfm, 5, iter.max = 100, nstart = 50, algorithm = 'Lloyd')
s5 <- plot(silhouette(k5$cluster, dist(scaled_rfm, 'euclidean')))

k6 <- kmeans(scaled_rfm, 6, iter.max = 100, nstart = 50, algorithm = 'Lloyd')
s6 <- plot(silhouette(k6$cluster, dist(scaled_rfm, 'euclidean')))

k7 <- kmeans(scaled_rfm, 7, iter.max = 100, nstart = 50, algorithm = 'Lloyd')
s7 <- plot(silhouette(k7$cluster, dist(scaled_rfm, 'euclidean')))

k8 <- kmeans(scaled_rfm, 8, iter.max = 100, nstart = 50, algorithm = 'Lloyd')
s8 <- plot(silhouette(k8$cluster, dist(scaled_rfm, 'euclidean')))

k9 <- kmeans(scaled_rfm, 9, iter.max = 100, nstart = 50, algorithm = 'Lloyd')
s9 <- plot(silhouette(k9$cluster, dist(scaled_rfm, 'euclidean')))

k10 <- kmeans(scaled_rfm, 10, iter.max = 100, nstart = 50, algorithm = 'Lloyd')
s10 <- plot(silhouette(k10$cluster, dist(scaled_rfm, 'euclidean')))



fviz_nbclust(scaled_rfm,kmeans,method = 'silhouette')

nb_results <- NbClust(scaled_rfm, diss = NULL, distance = 'euclidean', min.nc = 2, max.nc = 10,
                      method = 'kmeans', index = 'silhouette')

fviz_nbclust(nb_results, geom = "point", show.clustvar = FALSE)

#gap statistic method
set.seed(125)
stat_gap <- clusGap(scaled_rfm, FUN = kmeans, nstart = 25, K.max = 10)

fviz_gap_stat(stat_gap)


# Optimal is 4

### PCA Method
iss <- function(k) {
  kmeans(pca_data, k, iter.max = 100, nstart = 100, algorithm = 'Lloyd')$tot.withinss
}

k.values <- 1:10

iss_values <- map_dbl(k.values, iss)

plot(k.values, iss_values,
     type = 'b', pch = 19, frame = FALSE,
     xlab = "Number of clusters K",
     ylab = "Total intra-clusters sum of squares")

# Average silhouette method



k2 <- kmeans(pca_data, 2, iter.max = 100, nstart = 50, algorithm = 'Lloyd')
s2 <- plot(silhouette(k2$cluster, dist(pca_data, 'euclidean')))

k3 <- kmeans(pca_data, 3, iter.max = 100, nstart = 50, algorithm = 'Lloyd')
s3 <- plot(silhouette(k3$cluster, dist(pca_data, 'euclidean')))

k4 <- kmeans(pca_data, 4, iter.max = 100, nstart = 50, algorithm = 'Lloyd')
s4 <- plot(silhouette(k4$cluster, dist(pca_data, 'euclidean')))

k5 <- kmeans(pca_data, 5, iter.max = 100, nstart = 50, algorithm = 'Lloyd')
s5 <- plot(silhouette(k5$cluster, dist(pca_data, 'euclidean')))

k6 <- kmeans(pca_data, 6, iter.max = 100, nstart = 50, algorithm = 'Lloyd')
s6 <- plot(silhouette(k6$cluster, dist(pca_data, 'euclidean')))

k7 <- kmeans(pca_data, 7, iter.max = 100, nstart = 50, algorithm = 'Lloyd')
s7 <- plot(silhouette(k7$cluster, dist(pca_data, 'euclidean')))

k8 <- kmeans(pca_data, 8, iter.max = 100, nstart = 50, algorithm = 'Lloyd')
s8 <- plot(silhouette(k8$cluster, dist(pca_data, 'euclidean')))

k9 <- kmeans(pca_data, 9, iter.max = 100, nstart = 50, algorithm = 'Lloyd')
s9 <- plot(silhouette(k9$cluster, dist(pca_data, 'euclidean')))

k10 <- kmeans(pca_data, 10, iter.max = 100, nstart = 50, algorithm = 'Lloyd')
s10 <- plot(silhouette(k10$cluster, dist(pca_data, 'euclidean')))

# NbClust package



fviz_nbclust(pca_data,kmeans,method = 'silhouette')

nb_results <- NbClust(pca_data, diss = NULL, distance = 'euclidean', min.nc = 2, max.nc = 10,
                      method = 'kmeans', index = 'silhouette')

fviz_nbclust(nb_results, geom = "point", show.clustvar = FALSE)

# Gap statistic method



set.seed(125)
stat_gap <- clusGap(pca_data, FUN = kmeans, nstart = 25, K.max = 10)

fviz_gap_stat(stat_gap)

#Optimal is 2

##### 4 Creating Clusters

kmeans_result_all <- kmeans(scaled_data, 2, nstart = 25)
customer_clusters_all <- kmeans_result_all$cluster
cluster_data$cluster1 <- customer_clusters_all


kmeans_result_rfm <- kmeans(scaled_rfm, 4, nstart = 25)
customer_clusters_rfm <- kmeans_result_rfm$cluster
cluster_data$cluster2 <- customer_clusters_rfm

kmeans_result_pca <- kmeans(pca_data, 2, nstart = 25)
customer_clusters_pca <- kmeans_result_pca$cluster
cluster_data$cluster3 <- customer_clusters_pca


########5 Summarize Clusters 

#All
cluster_summary_all <- cluster_data %>%
  group_by(cluster1) %>%
  summarize(
    avg_total_sales = mean(total_sales),
    avg_total_profit = mean(total_profit),
    avg_total_quantity = mean(total_quantity),
    avg_total_orders = mean(total_orders),
    avg_recency = mean(recency),
    avg_days_as_customer = mean(days_as_customer),
    cluster_size = n()
  )

#RFM
cluster_summary_rfm <- cluster_data %>%
  group_by(cluster2) %>%
  summarize(
    avg_total_sales = mean(total_sales),
    avg_total_profit = mean(total_profit),
    avg_total_quantity = mean(total_quantity),
    avg_total_orders = mean(total_orders),
    avg_recency = mean(recency),
    avg_days_as_customer = mean(days_as_customer),
    cluster_size = n()
  )

#PCA
cluster_summary_pca <- cluster_data %>%
  group_by(cluster3) %>%
  summarize(
    avg_total_sales = mean(total_sales),
    avg_total_profit = mean(total_profit),
    avg_total_quantity = mean(total_quantity),
    avg_total_orders = mean(total_orders),
    avg_recency = mean(recency),
    avg_days_as_customer = mean(days_as_customer),
    cluster_size = n()
  )
print(cluster_summary_all)
print(cluster_summary_rfm)
print(cluster_summary_pca)

###NOTE SOME OF THESE CLUSTER MAY SHIFT AND CHANGE POSITIONS UPON RUNNING

#RFM looks the best creates four segments
#1.Lapsed Customers: This customer group orders the least frequently and has not ordered in a long time
#but there is some profitability there. They might be sensitive to a discount or a winback campaign
#They average around $199 of profit per customer

#2. Discount Only Shopper: This Customer segment  is the least profitable. They order more frequently and more 
#recently than the lapsed customer but they buy discounted products or low profit margin products. 
#They are on the hunt for a deal and will not spend that much. They are also the newest acquired customers.

#3.Top Customers: Most profitable customers They don’t order the most frequently but when they do it is large orders 
#with big profit margins. These are the customers we want to keep happy and shopping with us because they are very profitable.

#4.Rising Top Customers: This customer shopped the most recently and orders the most frequently, they are the 
#second most profitable. Can we get them to place larger orders and push them into the top customer tier


plot <- ggplot(data = cluster_data, aes(x = total_profit, y = total_orders, z = recency, color = factor(cluster2))) +
  geom_point() +
  labs(title = "Clusters") +
  theme_minimal()
