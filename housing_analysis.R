# Load required packages
library(tidyverse)
library(skimr)

# Load the dataset
housing_data <- read.csv("C:\\Users\\DELL\\Documents\\favoured\\datasets\\nigeria_houses_data.csv")

# Summary statistics
summary(housing_data)
skim(housing_data)  # Provides a detailed summary with histograms

# Check for missing values
colSums(is.na(housing_data))

# Outlier detection for price using IQR method
Q1 <- quantile(housing_data$price, 0.25, na.rm = TRUE)
Q3 <- quantile(housing_data$price, 0.75, na.rm = TRUE)
IQR <- Q3 - Q1
outliers <- housing_data$price < (Q1 - 1.5 * IQR) | housing_data$price > (Q3 + 1.5 * IQR)
housing_data[outliers, ]

# Encode categorical variables (e.g., title, state)
housing_data$title <- as.factor(housing_data$title)
housing_data$state <- as.factor(housing_data$state)
housing_data$town <- as.factor(housing_data$town)

# Normalize numerical variables (e.g., price)
housing_data$price_scaled <- scale(housing_data$price)

# Correlation matrix for numerical variables
cor_matrix <- cor(housing_data[, c("bedrooms", "bathrooms", "toilets", "parking_space", "price")], use = "complete.obs")
print(cor_matrix)

# Visualize correlation matrix
library(corrplot)
corrplot(cor_matrix, method = "color")

# ANOVA to compare price across property types
anova_result <- aov(price ~ title, data = housing_data)
summary(anova_result)

# Linear regression to predict price
lm_model <- lm(price ~ bedrooms + bathrooms + toilets + parking_space + title + state, data = housing_data)
summary(lm_model)

reg_predictions <- predict(lm_model, housing_data)
final_data <- cbind(Actual = housing_data$price, Predicted =reg_predictions)
final_data <- as.data.frame(final_data)

ggplot(data.frame(actual = housing_data$price, predicted = reg_predictions),
       aes(x = actual, y = predicted)) + 
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = 'red') +
  scale_x_log10() + scale_y_log10() +
  labs(title = 'Linear Regression: Predicted vs Actual Prices', 
       x = "Actual Price (NGN)", y = "Predicted Price (NGN)")

# Load visualization packages
library(ggplot2)

# Histogram of price
ggplot(housing_data, aes(x = price)) +
  geom_histogram(bins = 30, fill = "blue", alpha = 0.7) +
  scale_x_log10() +  # Log scale for price due to wide range
  theme_minimal() +
  labs(title = "Distribution of House Prices", x = "Price (NGN)", y = "Count")

# Boxplot of price by property type
ggplot(housing_data, aes(x = title, y = price)) +
  geom_boxplot(fill = "lightgreen") +
  scale_y_log10() +
  theme_minimal() +
  labs(title = "House Prices by Property Type", x = "Property Type", y = "Price (NGN)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Bar plot of average price by state
avg_price_state <- housing_data %>%
  group_by(state) %>%
  summarise(avg_price = mean(price, na.rm = TRUE))

ggplot(avg_price_state, aes(x = reorder(state, -avg_price), y = avg_price)) +
  geom_bar(stat = "identity", fill = "purple") +
  theme_minimal() +
  labs(title = "Average House Price by State", x = "State", y = "Average Price (NGN)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Scatter plot of price vs. bedrooms
ggplot(housing_data, aes(x = bedrooms, y = price)) +
  geom_point(color = "darkblue", alpha = 0.5) +
  scale_y_log10() +
  theme_minimal() +
  labs(title = "Price vs. Number of Bedrooms", x = "Bedrooms", y = "Price (NGN)")

# Load machine learning packages
library(caret)
library(randomForest)
library(xgboost)

# Prepare data for machine learning (remove rows with missing values)
housing_data_clean <- na.omit(housing_data)

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(housing_data_clean$price, p = 0.8, list = FALSE)
train_data <- housing_data_clean[trainIndex, ]
test_data <- housing_data_clean[-trainIndex, ]

# Random Forest for price prediction
rf_model <- randomForest(price ~ bedrooms + bathrooms + toilets + parking_space + title + state, 
                         data = train_data, ntree = 100)
predictions <- predict(rf_model, test_data)
final_data1 <- cbind(Actual = test_data$price, Predicted =predictions)
final_data1 <- as.data.frame(final_data1)

rmse <- sqrt(mean((predictions - test_data$price)^2))
print(paste("RMSE:", rmse))

# Feature importance
varImpPlot(rf_model)

# XGBoost for price prediction
xgb_data <- model.matrix(price ~ bedrooms + bathrooms + toilets + parking_space + title + state, 
                         data = train_data)[, -1]
xgb_label <- train_data$price
xgb_model <- xgboost(data = xgb_data, label = xgb_label, nrounds = 100, objective = "reg:squarederror")
xgb_pred <- predict(xgb_model, model.matrix(price ~ bedrooms + bathrooms + toilets + parking_space + title + state, 
                                            data = test_data)[, -1])
xgb_rmse <- sqrt(mean((xgb_pred - test_data$price)^2))
print(paste("XGBoost RMSE:", xgb_rmse))

# K-means clustering
set.seed(123)
kmeans_data <- housing_data_clean[, c("bedrooms", "bathrooms", "toilets", "parking_space", "price_scaled")]
kmeans_result <- kmeans(kmeans_data, centers = 3)
housing_data_clean$cluster <- kmeans_result$cluster

# Visualize clusters
ggplot(housing_data_clean, aes(x = bedrooms, y = price, color = as.factor(cluster))) +
  geom_point() +
  theme_minimal() +
  labs(title = "K-means Clustering of Housing Data", x = "Bedrooms", y = "Price (NGN)")

