# Load required libraries
library(ggplot2)
library(dplyr)
library(caret)
library(e1071)
library(corrplot)
library(readr)
library(caTools)
library(randomForest)
library(ROCR)
library(pROC)
library(nnet)

# Load the dataset
df <- read_csv("tripadvisor_review.csv")

# View summary of the dataset
summary(df)

# Check for missing values
sum(is.na(df))

# Data Preprocessing - Normalization
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

df_normalized <- as.data.frame(lapply(df[, -1], normalize))

df$Category <- as.factor(df$Category)  # Convert categorical variable to factor

# Splitting the dataset into Training and Test sets
set.seed(123)
split <- sample.split(df$Category, SplitRatio = 0.8)
training_set <- subset(df, split == TRUE)
test_set <- subset(df, split == FALSE)

# Exploratory Data Analysis
# Correlation Matrix
cor_matrix <- cor(df_normalized)
corrplot(cor_matrix, method="color", type="upper", tl.cex = 0.7)

# Distribution of Ratings
ggplot(df, aes(x = Category)) +
  geom_bar(fill = "steelblue") +
  theme_minimal() +
  labs(title = "Distribution of Ratings", x = "Categories", y = "Count")

# Boxplot to check for outliers
ggplot(df, aes(x = Category, y = df_normalized[,1])) +
  geom_boxplot(fill = "lightblue") +
  theme_minimal() +
  labs(title = "Boxplot of Ratings by Category", x = "Category", y = "Ratings")

# Model Implementation
# Support Vector Machine (SVM)
svm_model <- svm(Category ~ ., data = training_set, kernel = "linear", probability = TRUE)
svm_predictions <- predict(svm_model, test_set, probability = TRUE)
confusionMatrix(svm_predictions, test_set$Category)

# Logistic Regression Model
logistic_model <- train(Category ~ ., data = training_set, method = "glm", family = "binomial")
logistic_predictions <- predict(logistic_model, test_set)
confusionMatrix(logistic_predictions, test_set$Category)

# Random Forest Model
rf_model <- randomForest(Category ~ ., data = training_set, ntree = 100)
rf_predictions <- predict(rf_model, test_set)
confusionMatrix(rf_predictions, test_set$Category)

# Neural Network Model
nn_model <- nnet(Category ~ ., data = training_set, size = 5, maxit = 200)
nn_predictions <- predict(nn_model, test_set, type = "class")
confusionMatrix(nn_predictions, test_set$Category)

# Model Evaluation
# Accuracy Comparison
svm_accuracy <- mean(svm_predictions == test_set$Category)
logistic_accuracy <- mean(logistic_predictions == test_set$Category)
rf_accuracy <- mean(rf_predictions == test_set$Category)
nn_accuracy <- mean(nn_predictions == test_set$Category)

accuracy_df <- data.frame(Model = c("SVM", "Logistic Regression", "Random Forest", "Neural Network"), 
                          Accuracy = c(svm_accuracy, logistic_accuracy, rf_accuracy, nn_accuracy))

print(accuracy_df)

# Visualizing Model Performance
ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model Accuracy Comparison", y = "Accuracy", x = "Model")

# ROC Curve Comparison
svm_prob <- attributes(predict(svm_model, test_set, probability = TRUE))$probabilities[,2]
logistic_prob <- predict(logistic_model, test_set, type = "prob")[,2]
rf_prob <- predict(rf_model, test_set, type = "prob")[,2]

roc_svm <- roc(test_set$Category, svm_prob)
roc_logistic <- roc(test_set$Category, logistic_prob)
roc_rf <- roc(test_set$Category, rf_prob)

plot(roc_svm, col = "blue", main = "ROC Curve Comparison")
lines(roc_logistic, col = "red")
lines(roc_rf, col = "green")
legend("bottomright", legend = c("SVM", "Logistic Regression", "Random Forest"), col = c("blue", "red", "green"), lwd = 2)
