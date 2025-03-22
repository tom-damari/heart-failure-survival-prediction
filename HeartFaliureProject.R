# Libraries
library('plyr')
library('ggplot2')
library('pROC')
library('interactions')
library('glmnet')
library('MASS')
library('caret')


# Upload the data set
train_data <- read.csv(file.choose(),header=T)
test_data <- read.csv(file.choose(),header=T)

############################################ Interactions ############################################

####### define variables as factorial ########

train_data$smoking <- as.factor(train_data$smoking)
train_data$sex <- as.factor(train_data$sex)
train_data$diabetes <- as.factor(train_data$diabetes)


####### interaction plots ########

# Plot interaction between sex and smoking with DEATH_EVENT
ggplot(train_data, aes(x = sex, fill = smoking)) +
  geom_bar(position = "dodge") +
  geom_text(stat = "count", aes(label = ..count..), vjust = 1, position = position_dodge(width = 1)) +  # Adjust vjust and position to position count labels above bars
  facet_grid(DEATH_EVENT ~ ., labeller = label_both) +
  theme(strip.text.x.top = element_text(hjust = 0.5)) +
  labs(title = "Interaction between Sex and Smoking by DEATH_EVENT",
       x = "Sex",
       y = "Count",
       fill = "Smoking") + 
  scale_fill_manual(values = c("0" = "skyblue", "1" = "#FFB6C1")) + 
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA),
        plot.margin = margin(20, 20, 50, 20)) 

# Plot interaction between diabetes and smoking with DEATH_EVENT
ggplot(train_data, aes(x = diabetes, fill = smoking)) +
  geom_bar(position = "dodge") +
  geom_text(stat = "count", aes(label = ..count..), vjust = 1, position = position_dodge(width = 1)) +  # Adjust vjust and position to position count labels above bars
  facet_grid(DEATH_EVENT ~ ., labeller = label_both) +
  theme(strip.text.x.top = element_text(hjust = 0.5)) +
  labs(title = "Interaction between Diabetes and Smoking by DEATH_EVENT",
       x = "Diabetes",
       y = "Count",
       fill = "Smoking") + 
  scale_fill_manual(values = c("0" = "skyblue", "1" = "#FFB6C1")) + 
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA),
        plot.margin = margin(20, 20, 50, 20)) 


############################################ Logistic regression model ############################################

# Function to generate equation string from a model
generate_equation_string <- function(model) {
  coef_vector <- coef(model)
  equation_str <- "log(p / (1 - p)) = "
  for (i in seq_along(coef_vector)) {
    coef_val <- coef_vector[i]
    if (i == 1) {
      equation_str <- paste0(equation_str, round(coef_val, 2))
    } else {
      variable_name <- names(coef_vector)[i]
      if (coef_val >= 0) {
        sign_str <- " + "
      } else {
        sign_str <- " - "
        coef_val <- abs(coef_val)  # Make coefficient positive for display
      }
      equation_str <- paste0(equation_str, sign_str, "(", round(coef_val, 2), " * ", variable_name, ")")
    }
  }
  return(equation_str)
}


####### full model ########
initialModel <- glm(DEATH_EVENT ~ age + factor(anaemia) + creatinine_phosphokinase + factor(diabetes) + ejection_fraction + factor(high_blood_pressure) + platelets + serum_creatinine + factor(sex) + factor(smoking) + factor(time_in_months) + factor(sex)*factor(smoking) + factor(diabetes)*factor(smoking) , data = train_data, family = binomial(link = "logit"))
summary(initialModel)
BIC(initialModel)


####### Models ########

# Variables selection
fullModel <- initialModel
clearModel <- glm(DEATH_EVENT ~ 1 , data = train_data, family = binomial(link = "logit"))

#### 1. Backward Elimination ####
beModel <- step(fullModel, direction = 'backward')
summary(beModel)
AIC_be <- AIC(beModel)
BIC_be <- BIC(beModel)
print(paste("AIC for Backward Elimination Model:", round(AIC_be, 3)))
print(paste("BIC for Backward Elimination Model:", round(BIC_be, 3)))

# Equation of the chosen model
be_selected_features <- names(coef(beModel))
be_equation_str <- generate_equation_string(beModel)
print("Backward Elimination Model Equation:")
print(be_equation_str)

#AUC-RUC
predictions_be <- predict(beModel, newdata=test_data, type="response")
roc_be <- roc(test_data$DEATH_EVENT, predictions_be)
auc_be <- auc(roc_be)
print(paste("AUC for Backward Elimination Model:", round(auc_be, 4)))

#plot
plot(roc_be, main="ROC Curve for Backward Elimination Model")


#### 2. Forward selection ####

fsModel <- step(clearModel, direction = 'forward', scope = ~ age + factor(anaemia) + creatinine_phosphokinase + factor(diabetes) + ejection_fraction + factor(high_blood_pressure) + platelets + serum_creatinine + factor(sex) + factor(smoking) + factor(time_in_months) + factor(sex)*factor(smoking) + factor(diabetes)*factor(smoking),data = train_data)
summary(fsModel)

# AIC and BIC for the forward selection model
AIC_fs <- AIC(fsModel)
BIC_fs <- BIC(fsModel)
print(paste("AIC for Forward Selection Model:", round(AIC_fs, 3)))
print(paste("BIC for Forward Selection Model:", round(BIC_fs, 3)))

# Equation of the chosen model
selected_features <- names(coef(fsModel))
fs_equation_str <- generate_equation_string(fsModel)
print("Forward Selection Model Equation:")
print(fs_equation_str)

# AUC-ROC test score
predictions_fs <- predict(fsModel, newdata = test_data, type = "response")
roc_fs <- roc(test_data$DEATH_EVENT, predictions_fs)
auc_fs <- auc(roc_fs)
print(paste("AUC for Forward Selection Model:", round(auc_fs,4)))

#### 3. Stepwise Selection ####
ssModel <- step(initialModel, direction = 'both')
summary(ssModel)

AIC_ss <- AIC(ssModel)
BIC_ss <- BIC(ssModel)
print(paste("AIC for Stepwise Selection Model:", round(AIC_ss, 3)))
print(paste("BIC for Stepwise Selection Model:", round(BIC_ss, 3)))

# Equation of the chosen model
selected_features <- names(coef(ssModel))
ss_equation_str <- generate_equation_string(ssModel)
print("Stepwise Selection Model Equation:")
print(ss_equation_str)

# AUC-ROC test score
predictions_ss <- predict(ssModel, newdata = test_data, type = "response")
roc_ss <- roc(test_data$DEATH_EVENT, predictions_ss)
auc_ss <- auc(roc_ss)
print(paste("AUC for Stepwise Selection Model:", round(auc_ss,4)))


#### 4. Lasso Elimination ####

# define factorial variables - train set
train_data$anaemia <- as.factor(train_data$anaemia)
train_data$diabetes <- as.factor(train_data$diabetes)
train_data$high_blood_pressure <- as.factor(train_data$high_blood_pressure)
train_data$sex <- as.factor(train_data$sex)
train_data$smoking <- as.factor(train_data$smoking)
train_data$time_in_months <- as.factor(train_data$time_in_months)

# Define factor variables and convert them into dummy variables
factor_vars <- c("anaemia", "diabetes", "high_blood_pressure", "sex", "smoking", "time_in_months")

train_data_dummies <- model.matrix(~.-1, data = train_data[, -which(names(train_data) == "DEATH_EVENT")], 
                                   contrasts.arg = lapply(train_data[factor_vars], contrasts, contrasts = TRUE))

# Calculate the number of levels for each factor variable
num_levels <- sapply(train_data[factor_vars], function(x) length(unique(x)))
print(num_levels)

# Specify contrasts argument to define base level for anaemia
train_data_dummies <- model.matrix(~.-1, data = train_data[, -which(names(train_data) == "DEATH_EVENT")], 
                                   contrasts.arg = list(
                                     anaemia = contr.treatment(n = num_levels["anaemia"]),
                                     diabetes = contr.treatment(n = num_levels["diabetes"]),
                                     high_blood_pressure = contr.treatment(n = num_levels["high_blood_pressure"]),
                                     sex = contr.treatment(n = num_levels["sex"]),
                                     smoking = contr.treatment(n = num_levels["smoking"]),
                                     time_in_months = contr.treatment(n = num_levels["time_in_months"])
                                   ))

# Calculate the number of levels for each factor variable


# Split test and train sets
X_train <- as.matrix(train_data_dummies)
y_train <- train_data$DEATH_EVENT

# Perform Lasso elimination
lasso_model_final <- cv.glmnet(X_train, y_train, family = "binomial")
# Get the lambda with the minimum cross-validated deviance
best_lambda <- lasso_model_final$lambda.min  

# Plot lambda values versus cross-validated deviance
plot(lasso_model_final$lambda, lasso_model_final$cvm, type = "b", pch = 19, col = "blue",
     xlab = "Lambda", ylab = "Cross-Validated Deviance",
     main = "Lambda vs Cross-Validated Deviance for Logistic Regression with Lasso Regularization")
# Add the best lambda value as a red point
best_lambda_index <- which.min(lasso_model_final$cvm)
best_lambda <- lasso_model_final$lambda[best_lambda_index]
points(best_lambda, lasso_model_final$cvm[best_lambda_index], col = "red", pch = 19)
# Add a vertical line at the best lambda value
abline(v = best_lambda, col = "red", lty = 2)
print(best_lambda)

# Fit the chosen model with the selected lambda
lasso_model <- glmnet(X_train, y_train, family = "binomial", alpha = 1, lambda = best_lambda)
summary(lasso_model)

# Get the deviance of the chosen model
deviance <- deviance(lasso_model)
# Number of parameters in the model
num_params <- sum(coef(lasso_model, s = best_lambda) != 0)
# Compute AIC and BIC
n <- nrow(X_train)
AIC_lasso_model <- deviance + 2 * num_params
BIC_lasso_model <- deviance + log(n) * num_params
# Print AIC and BIC
print(paste("AIC for Chosen Lasso Model:", round(AIC_lasso_model, 3)))
print(paste("BIC for Chosen Lasso Model:", round(BIC_lasso_model, 3)))

print(coef(lasso_model))

# Extract coefficients and variable names for the chosen model
lasso_coefs <- as.vector(coef(lasso_model, s = best_lambda))
non_zero_coefs <- lasso_coefs[lasso_coefs != 0]
non_zero_variables <- colnames(X_train)[which(lasso_coefs != 0)]
intercept_index <- which(is.na(non_zero_variables))
if (length(intercept_index) > 0) {
  intercept_coef <- non_zero_coefs[intercept_index]
  non_zero_variables <- non_zero_variables[!is.na(non_zero_variables)]
  non_zero_coefs <- non_zero_coefs[-intercept_index]
} else {
  intercept_coef <- NA
}

print(intercept_coef)
print(non_zero_coefs)
print(non_zero_variables)

# Construct the equation model string with intercept
equation_str <- paste("Chosen Lasso Model Equation: log(p / (1 - p)) =", round(intercept_coef, 3), "+")
coefficients <- paste(round(non_zero_coefs, 2), "*", non_zero_variables)
equation_str <- paste(equation_str, paste(coefficients, collapse = " + "))
print(equation_str)


# define factorial variables - test set
test_data$anaemia <- as.factor(test_data$anaemia)
test_data$diabetes <- as.factor(test_data$diabetes)
test_data$high_blood_pressure <- as.factor(test_data$high_blood_pressure)
test_data$sex <- as.factor(test_data$sex)
test_data$smoking <- as.factor(test_data$smoking)
test_data$time_in_months <- as.factor(test_data$time_in_months)



# Get the indices of non-zero coefficients
selected_indices <- which(lasso_coefs != 0)

# Extract selected variables and their dummy variables for test set
selected_vars <- non_zero_variables[selected_indices]
test_data_dummies_selected <- test_data_dummies[, selected_indices]

# Predictions with Lasso Model using selected variables
predictions_lasso_model <- predict(lasso_model, newx = test_data_dummies_selected, s = best_lambda, type = "response")

# Calculate AUC-ROC score
roc_auc_lasso <- roc(y_test, predictions_lasso_model)
auc_score_lasso <- auc(roc_auc_lasso)

# Print AUC-ROC score
print(paste("AUC-ROC Score for chosen Lasso Model:", round(auc_score_lasso, 4)))





# AUC-RUC
test_data_dummies <- model.matrix(~.-1, data = test_data[, -which(names(test_data) == "DEATH_EVENT")], 
                                  contrasts.arg = lapply(test_data[factor_vars], contrasts, contrasts = FALSE))
X_test <- as.matrix(test_data_dummies)
predictions_lasso_model <- predict(lasso_model, newx = X_test, s = best_lambda, type = "response")
roc_auc_lasso <- roc(y_test, predictions_lasso_model)
auc_score_lasso <- auc(roc_auc_lasso)
print(paste("AUC-ROC Score for chosen Lasso Model:", round(auc_score_lasso, 4)))



