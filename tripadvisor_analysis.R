# install.packages(c("tidyverse", "caret", "e1071", "nnet", "corrplot"))

# Load libraries
library(tidyverse)
install.packages("tidyverse")
library(caret)
library(e1071)
library(nnet)
library(corrplot)
install.packages("nnet")   # Only once
library(nnet)              # Every R session


data <- read.csv("C:\\Users\\Admin\\Desktop\\travel+reviews\\tripadvisor_review.csv")

data <- data[ , -1]

# Compute average across the 10 rating categories
avg_rating <- rowMeans(data)

# Create rating bins
data$Overall_Rating <- cut(
  avg_rating,
  breaks = c(1.2, 1.6, 1.85, 2.25),
  labels = c("1", "2", "3"),
  right = TRUE,
  include.lowest = TRUE
)

# Convert to factor
data$Overall_Rating <- as.factor(data$Overall_Rating)

# Check outputs
table(data$Overall_Rating)
summary(avg_rating)

table(data$Overall_Rating)

library(caret)

# Re-scale the 10 feature columns only (1:10)
features_scaled <- scale(data[ , 1:10])

# Combine scaled features + Overall_Rating column
data_scaled <- as.data.frame(cbind(features_scaled, Overall_Rating = data$Overall_Rating))

# Make sure target is still a factor
data_scaled$Overall_Rating <- as.factor(data_scaled$Overall_Rating)

# Confirm structure
str(data_scaled)

library(caret)

set.seed(42)

# Stratified sampling by class
train_index <- createDataPartition(data_scaled$Overall_Rating, p = 0.8, list = FALSE)

# Train and test sets
train_data <- data_scaled[train_index, ]
test_data <- data_scaled[-train_index, ]

# Check class balance
table(train_data$Overall_Rating)
table(test_data$Overall_Rating)

library(nnet)

# Train the model
log_model <- multinom(Overall_Rating ~ ., data = train_data)

# Predict on the test set
log_preds <- predict(log_model, newdata = test_data)

# Evaluate with confusion matrix
confusionMatrix(log_preds, test_data$Overall_Rating)

# Predict on the test set
log_preds <- predict(log_model, newdata = test_data)

# Evaluate with confusion matrix
confusionMatrix(log_preds, test_data$Overall_Rating)

library(e1071)

# Train SVM with radial kernel
svm_model <- svm(Overall_Rating ~ ., data = train_data, kernel = "radial")

# Predict
svm_preds <- predict(svm_model, newdata = test_data)

# Align factor levels
svm_preds <- factor(svm_preds, levels = c("1", "2", "3"))
test_labels <- factor(test_data$Overall_Rating, levels = c("1", "2", "3"))

# Evaluate
confusionMatrix(svm_preds, test_labels)

# Tune SVM parameters
set.seed(42)
tuned_svm <- tune(
  svm,
  Overall_Rating ~ .,
  data = train_data,
  kernel = "radial",
  ranges = list(cost = c(0.1, 1, 10), gamma = c(0.01, 0.1, 1))
)

# Best model
best_svm <- tuned_svm$best.model

# Predict on test set
svm_tuned_preds <- predict(best_svm, newdata = test_data)

# Evaluate
svm_tuned_preds <- factor(svm_tuned_preds, levels = c("1", "2", "3"))
confusionMatrix(svm_tuned_preds, test_data$Overall_Rating)

install.packages("randomForest")
library(randomForest)

# Set seed for reproducibility
set.seed(123)

# Train Random Forest
rf_model <- randomForest(
  Overall_Rating ~ .,
  data = train_data_smote,
  ntree = 500,        # number of trees
  mtry = 3,           # number of variables tried at each split (can tune)
  importance = TRUE   # track feature importance
)
install.packages("smotefamily")
library(smotefamily)

# Prepare training data
train_data_numeric <- train_data
train_data_numeric$Overall_Rating <- as.numeric(as.character(train_data$Overall_Rating))

# Separate features and target
X <- train_data_numeric[ , 1:10]              # features
y <- train_data_numeric$Overall_Rating        # target as numeric

# Apply SMOTE
library(smotefamily)
smote_output <- SMOTE(X, y, K = 5, dup_size = 2)  # 200% oversampling

# Combine features + class
train_data_smote <- smote_output$data
train_data_smote$Overall_Rating <- as.factor(round(train_data_smote$class))
train_data_smote$class <- NULL

# Check class balance
table(train_data_smote$Overall_Rating)

# Convert class column to numeric before rounding
train_data_smote$class <- as.numeric(as.character(train_data_smote$class))

# Round and convert to factor
train_data_smote$Overall_Rating <- as.factor(round(train_data_smote$class))

# Drop the old class column
train_data_smote$class <- NULL

# Check class balance
table(train_data_smote$Overall_Rating)

# Load Random Forest library
install.packages("randomForest")  # if not already installed
library(randomForest)

# Set seed and train the model
set.seed(123)
rf_model <- randomForest(
  Overall_Rating ~ .,
  data = train_data_smote,
  ntree = 500,        # number of trees
  mtry = 3,           # number of variables tried at each split
  importance = TRUE   # enables variable importance plotting
)

# Predict on test set
rf_preds <- predict(rf_model, newdata = test_data)

# Match factor levels
rf_preds <- factor(rf_preds, levels = c("1", "2", "3"))
test_labels <- factor(test_data$Overall_Rating, levels = c("1", "2", "3"))

# Evaluate
library(caret)
confusionMatrix(rf_preds, test_labels)

# Load library
library(ggplot2)

# Create a data frame with model comparison
model_results <- data.frame(
  Model = c("Logistic Regression", "Tuned SVM", "Random Forest (SMOTE)"),
  Accuracy = c(99.49, 96.92, 90.26)
)

# Plot
ggplot(model_results, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = paste0(Accuracy, "%")), vjust = -0.5, size = 4.5) +
  ylim(0, 105) +
  theme_minimal() +
  labs(title = "Model Accuracy Comparison",
       y = "Accuracy (%)", x = "") +
  theme(legend.position = "none",
        text = element_text(size = 12),
        plot.title = element_text(hjust = 0.5, face = "bold"))


# Create confusion matrix
cm_log <- confusionMatrix(log_preds, test_data$Overall_Rating)

# Convert to table
cm_log_df <- as.data.frame(cm_log$table)

# Plot
ggplot(cm_log_df, aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "white") +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "#D6EAF8", high = "#1F618D") +
  theme_minimal() +
  labs(title = "Logistic Regression – Confusion Matrix",
       x = "Actual Class", y = "Predicted Class") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
