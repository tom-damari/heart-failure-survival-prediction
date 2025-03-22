import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind, shapiro, levene, mannwhitneyu, chi2
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, make_scorer, \
    roc_curve, auc
import matplotlib.colors as mcolors
from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import product


# import data
data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#creating separate DF for Death event and survived patients
deadOnly = data[(data.DEATH_EVENT == 1)]
survivedOnly = data[(data.DEATH_EVENT == 0)]

############################################ Descriptive Statistics for Quantitative Variables + density plots ############################################

quantitative_data = data[['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']]
statisticsInfo = quantitative_data.describe()
skewness = quantitative_data.skew()
statisticsInfo.loc['skewness'] = skewness
print(statisticsInfo)

# Density plots for continuous variables
def plot_density(data, variable, bins=30, color='blue'):
    plt.figure(figsize=(8, 6))
    sns.histplot(data[variable], kde=True, color=color, bins=bins, edgecolor='black')
    plt.title(f'Density Plot of {variable}')
    plt.xlabel(variable)
    plt.ylabel('Density')
    plt.show()

plot_density(data, 'creatinine_phosphokinase', bins=30, color='blue')
plot_density(data, 'ejection_fraction', bins=30, color='blue')
plot_density(data, 'platelets', bins=30, color='blue')
plot_density(data, 'serum_creatinine', bins=30, color='blue')
plot_density(data, 'serum_sodium', bins=30, color='blue')

############################################ Frequencies & Donut Chart Plots - Categorical ############################################

anaemia_counts = data['anaemia'].value_counts()
print("Category Variable Counts: " + str(anaemia_counts))
colors = ['#ffb6c1', '#99ff99']
labels = [f'No Anaemia ({anaemia_counts[0]})', f'Anaemia ({anaemia_counts[1]})']
plt.pie(anaemia_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops=dict(width=0.3))
plt.title('Distribution of anaemia Variable')
plt.legend(labels, loc='upper right')
plt.show()

diabetes_counts = data['diabetes'].value_counts()
print("Category Variable Counts: " + str(diabetes_counts))
colors = ['#ffb6c1', '#99ff99']
labels = [f'No diabetes ({diabetes_counts[0]})', f'diabetes ({diabetes_counts[1]})']
plt.pie(diabetes_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops=dict(width=0.3))
plt.title('Distribution of diabetes Variable')
plt.legend(labels, loc='upper right')
plt.show()

high_bp_counts = data['high_blood_pressure'].value_counts()
print("Category Variable Counts: " + str(high_bp_counts))
colors = ['#ffb6c1', '#99ff99']
labels = [f'No high blood pressure ({high_bp_counts[0]})', f'high blood pressure ({high_bp_counts[1]})']
plt.pie(high_bp_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops=dict(width=0.3))
plt.title('Distribution of high_blood_pressure Variable')
plt.legend(labels, loc='upper right')
plt.show()

sex_counts = data['sex'].value_counts()
print("Category Variable Counts: " + str(sex_counts))
colors = ['#99ff99','#ffb6c1']
labels = [f'Man ({sex_counts[1]})', f'Woman ({sex_counts[0]})']
plt.pie(sex_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops=dict(width=0.3))
plt.title('Distribution of sex Variable')
plt.legend(labels, loc='upper right')
plt.show()

smoking_counts = data['smoking'].value_counts()
print("Category Variable Counts: " + str(smoking_counts))
colors = ['#ffb6c1', '#99ff99']
labels = [f'not smoking ({smoking_counts[0]})', f'smoking ({smoking_counts[1]})']
plt.pie(smoking_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops=dict(width=0.3))
plt.title('Distribution of smoking Variable')
plt.legend(labels, loc='upper right')
plt.show()

DEATH_EVENT_counts = data['DEATH_EVENT'].value_counts()
print("Category Variable Counts: " + str(DEATH_EVENT_counts))
colors = ['#ffb6c1', '#99ff99']
labels = [f'survived ({DEATH_EVENT_counts[0]})', f'died ({DEATH_EVENT_counts[1]})']
plt.pie(DEATH_EVENT_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops=dict(width=0.3))
plt.title('Distribution of DEATH_EVENT Variable')
plt.legend(labels, loc='upper right')
plt.show()

############################################ Boxplots & Outliers ############################################

plt.figure(figsize=(15, 10))
outlier_values = {}  # Dictionary to store outliers for each variable
for i, column in enumerate(quantitative_data.columns, 1):
    plt.subplot(3, 3, i)
    boxplot_result = plt.boxplot(quantitative_data[column])
    # Access and store outliers for each variable
    outliers = [item.get_ydata() for item in boxplot_result['fliers']]
    outlier_values[column] = outliers
    plt.title(f'Boxplot for {column}')
plt.tight_layout()
plt.show()
# Print outlier values for each variable
for column, outliers in outlier_values.items():
    print(f"Outliers for {column}: {outliers}")


############################################ Correlation between variables ############################################

#Calculate the correlation matrix between continuous variables
corr_matrix = quantitative_data.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5, ax=ax)
plt.title('Correlation Between continuous variables')
plt.show()

# Between 2 categorical variables
def barPlots(x_attribute, y_attribute):
    bar_plot_df = data[[x_attribute, y_attribute]]
    cross_tab_prop = pd.crosstab(index=bar_plot_df[x_attribute],
                                 columns=bar_plot_df[y_attribute],
                                 normalize="index")
    colors = ['#ffb6c1', '#99ff99']
    custom_cmap = mcolors.ListedColormap(colors)
    ax = cross_tab_prop.plot(kind='bar', stacked=True, colormap=custom_cmap, figsize=(10, 6))
    plt.legend(loc="upper left", ncol=2, title=y_attribute)
    plt.xlabel(x_attribute)
    plt.ylabel("Proportion")
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{height:.2%}', (x + width / 2, y + height / 2), ha='center', va='center')
    plt.show()

# Chi square test for 2 categorical variables
def chi_squared_test(data, variable1, variable2):
    contingency_table = pd.crosstab(data[variable1], data[variable2])
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-squared test between {variable1} and {variable2}:")
    print(f"Chi-squared statistic: {chi2_stat}")
    print(f"P-value: {p_value}")
    print(f"Degrees of freedom: {dof}")
    if p_value < 0.05:
        print("The result is statistically significant (p-value < 0.05).")
    else:
        print("The result is not statistically significant (p-value >= 0.05).")

    return chi2_stat, p_value, dof, expected

# Checking plots to see if there is correlation between 2 categorical variables
barPlots('sex', 'smoking')
barPlots('diabetes','high_blood_pressure')
barPlots('anaemia','sex')
barPlots('smoking', 'diabetes')
barPlots('anaemia','high_blood_pressure')
barPlots('anaemia', 'smoking')
barPlots('sex', 'high_blood_pressure')
barPlots('smoking', 'high_blood_pressure')
barPlots('anaemia','diabetes')

# chi test for interesting pairs
chi_squared_test(data,'sex','smoking')
chi_squared_test(data,'smoking','diabetes')


# Between categorical and continuous variables
def plot_boxplot(data, x_variable, y_variable):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=x_variable, y=y_variable, data=data, palette=['lightcoral', 'lightgreen'])
    plt.title(f'Box Plot: {y_variable} by {x_variable}', fontsize=16)
    plt.xlabel(x_variable, fontsize=14)
    plt.ylabel(y_variable, fontsize=14)
    plt.show()

# Checking boxplots to see if there is correlation between categorical variable and continuous variable
plot_boxplot(data, 'sex', 'age')
plot_boxplot(data, 'high_blood_pressure', 'age')
plot_boxplot(data, 'smoking', 'age')
plot_boxplot(data, 'diabetes', 'age')
plot_boxplot(data, 'diabetes', 'serum_creatinine')
plot_boxplot(data, 'high_blood_pressure', 'serum_creatinine')
plot_boxplot(data, 'anaemia', 'platelets')
plot_boxplot(data, 'high_blood_pressure', 'platelets')

############################################ Multicollinearity check for continuous variables ############################################

# Select only the columns with continuous variables
X = quantitative_data
X_with_constant = pd.concat([pd.Series(1, index=X.index, name='const'), X], axis=1)
vif_data = pd.DataFrame()
vif_data["Variable"] = X_with_constant.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_constant.values, i) for i in range(X_with_constant.shape[1])]
# Display the VIF results
vif_data = vif_data[vif_data['Variable'] != 'const']
print(vif_data)

############################################ Plots + frequencies (if changed into categorical) ############################################

# bar plot of age
data['age'] = data['age'].astype(int)
x_variable = 'age'  # Replace with your continuous variable
hue_variable = 'DEATH_EVENT'  # Replace with your binary response variable
plt.figure(figsize=(10, 6))
sns.countplot(x=x_variable, hue=hue_variable, data=data, palette={0: 'lightgreen', 1: 'lightcoral'})
plt.title(f'Bar Plot: {x_variable} and {hue_variable}', fontweight='bold', fontsize=20)
plt.xlabel(x_variable, fontweight='bold', fontsize=16)
plt.ylabel('Count', fontweight='bold',  fontsize=16)
plt.legend(title=hue_variable, title_fontsize='12', loc='upper right')
plt.xticks(fontsize=16)
plt.show()


# Create a density plot with bars for time divided by 'DEATH_EVENT'
plt.figure(figsize=(12, 8))
sns.histplot(data, x='time', hue='DEATH_EVENT', kde=True, bins=30, palette={0: 'lightgreen', 1: 'lightcoral'}, edgecolor='black')
plt.title('Density Plot of time by DEATH_EVENT', fontweight='bold', fontsize=20)
plt.xlabel('time', fontweight='bold', fontsize=16)
plt.ylabel('Density', fontweight='bold', fontsize=16)
legend_labels = ['0', '1']
legend_colors = ['lightgreen', 'lightcoral']
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]
plt.legend(legend_handles, legend_labels, title='DEATH_EVENT', title_fontsize='12', loc='upper right')
plt.show()

# Convert 'time' to months
data['time_in_months'] = (data['time'] // 30).astype(int)
dataForRegression = data.copy()

# Plot of time in months
plt.figure(figsize=(12, 8))
sns.histplot(data, x='time_in_months', hue='DEATH_EVENT', kde=True, bins=30, palette={0: 'lightgreen', 1: 'lightcoral'}, edgecolor='black')
plt.title('Density Plot of Time (Months) by DEATH_EVENT', fontweight='bold', fontsize=20)
plt.xlabel('Time (Months)', fontweight='bold', fontsize=16)
plt.ylabel('Density', fontweight='bold', fontsize=16)
legend_labels = ['Survived', 'Died']  # Adjusting labels for clarity
legend_colors = ['lightgreen', 'lightcoral']
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]
plt.legend(legend_handles, legend_labels, title='DEATH_EVENT', title_fontsize='12', loc='upper right')
plt.show()

############################################ Dummy variable + Normalization ############################################

# months dummies
month_dummies = pd.get_dummies(data['time_in_months'], prefix='month', dtype=int)
data = pd.concat([data, month_dummies], axis=1)
data = data.drop(['time', 'time_in_months'], axis=1)

# Preparing dataFrame for the upcoming Logistic regression
dataForRegression = dataForRegression.drop(['time'], axis=1)

# Move the 'DEATH EVENT' to the end of the data Frame
column_to_move = data.pop('DEATH_EVENT')  # remove the column from the data frame
data.insert(len(data.columns), 'DEATH_EVENT', column_to_move)  # insert the column at the end of the data frame

# Normalized continuous variables
columns_to_scale = ['age','creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']
scaler = StandardScaler()
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

# Preparing dataFrame for the upcoming Logistic regression
column_to_move = dataForRegression.pop('DEATH_EVENT')  # remove the column from the data frame
dataForRegression.insert(len(dataForRegression.columns), 'DEATH_EVENT', column_to_move)  # insert the column at the end of the data frame
dataForRegression[columns_to_scale] = scaler.fit_transform(dataForRegression[columns_to_scale])


############################################ Decision tree ############################################
y = data['DEATH_EVENT']
X = data.drop('DEATH_EVENT', axis=1)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# After splitting-building a full tree
firstModel = DecisionTreeClassifier(criterion='entropy',random_state=42)
firstModel.fit(X_train, y_train)
plt.figure(figsize=(30, 10))
plot_tree(firstModel, filled=True, feature_names=X_train.columns,class_names=True)
plt.show()
max_tree_depth = firstModel.tree_.max_depth
num_leaf_nodes = firstModel.tree_.n_leaves
print('max depth of first tree:'+str(max_tree_depth))
print('number of leaves in the first tree:'+str(num_leaf_nodes))
## AUC-ROC on the full and first tree
print(f"AUC-ROC Score full and first tree train set: {roc_auc_score(y_train, firstModel.predict_proba(X_train)[:, 1]):.4}")
print(f"AUC-ROC Score full and first tree test set: {roc_auc_score(y_test, firstModel.predict_proba(X_test)[:, 1]):.4}")

# Evaluation Method & Hyperparameter Tuning - Plots in order to understand the scale of the parameters
#GridsearchCV
param_grid = {
                'max_depth': np.arange(1,10, 1),
                'criterion': ['entropy', 'gini'],
                'max_features': np.arange(3,21, 1)
             }
comb = 1
for list_ in param_grid.values():
    comb *= len(list_)
print("number of Combinations in GridSearch:"+str(comb))
param_grid.values()

#Grid search
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                           param_grid=param_grid,
                           refit=True,
                           cv=10, scoring='roc_auc')

grid_search.fit(X_train, y_train)
best_modelGS = grid_search.best_estimator_
print('After grid_search the best model is:'+str(best_modelGS))
print(f"GridSearchCV- AUC-ROC Score on train set: {roc_auc_score(y_train, best_modelGS.predict_proba(X_train)[:, 1]):.4}")
print(f"GridSearchCV- AUC-ROC Score on test set: {roc_auc_score(y_test, best_modelGS.predict_proba(X_test)[:, 1]):.4}")


#RandomsearchCV
param_grid={'max_depth': np.arange(1,10, 1),
            'criterion': ['entropy', 'gini'],
            'max_features' : np.arange(3,21, 1),
            'min_samples_leaf':np.arange(5,15,1) }

comb = 1
for list_ in param_grid.values():
    comb *= len(list_)
print("number of Combinations in Randomsearch:"+str(comb))
param_grid.values()

random_search = RandomizedSearchCV(DecisionTreeClassifier(random_state=42),
                                   param_distributions=param_grid
                                   ,cv=10,random_state=123, n_iter=800, refit=True,scoring='roc_auc')


random_search.fit(X_train, y_train)
best_modelRS=random_search.best_estimator_
num_leaf_nodes1 = best_modelRS.tree_.n_leaves
print('number of leaves in the best_modelRS tree:'+str(num_leaf_nodes1))
print('After random_search the best model is:'+str(best_modelRS))
print(f"RandomizedSearchCV- AUC-ROC Score on train set : {roc_auc_score(y_train, best_modelRS.predict_proba(X_train)[:, 1]):.4}")
print(f"RandomizedSearchCV AUC-ROC Score on test set : {roc_auc_score(y_test, best_modelRS.predict_proba(X_test)[:, 1]):.4}")

# Printing the best decision tree we got - RandomSearch
plt.figure(figsize=(30,10))
plot_tree(best_modelRS, filled=True,feature_names=X_train.columns, class_names=True)
plt.show()

##feature importance
# Get the features importances
importance = best_modelRS.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})
# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)


# ############################################ Random forest ############################################

rf_classifier = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': [50,100,200,300],
    'max_depth': np.arange(5,15,1),
    'min_samples_split': np.arange(5,15,1),
}
random_search = RandomizedSearchCV(rf_classifier, param_distributions=param_dist, n_iter=10, cv=5, scoring='roc_auc', random_state=42)
random_search.fit(X_train, y_train)
best_model_RF = random_search.best_estimator_
# Print all best hyperparameters
selected_hyperparameters = ['n_estimators', 'max_depth', 'min_samples_split']
print("Best Hyperparameters in Random Forest after RandomSearchCV:")
for param in selected_hyperparameters:
    value = best_model_RF.get_params()[param]
    print(f"{param}: {value}")
print(f"RandomizedSearchCV Random Forest- AUC-ROC Score on train set : {roc_auc_score(y_train, best_model_RF.predict_proba(X_train)[:, 1]):.4}")
print(f"RandomizedSearchCV Random Forest- AUC-ROC Score on test set : {roc_auc_score(y_test, best_model_RF.predict_proba(X_test)[:, 1]):.4}")

# Confusion Matrix
y_pred = best_model_RF.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Compute ROC curve and AUC score
y_probs=best_model_RF.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

##feature importance
# Get the features importances
importance = best_model_RF.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})
# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)


################################################################## SVM #################################################################

#Grid Search
param_grid = {
    'C': np.arange(0.1, 1.2, 0.2),
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 0.01, 0.1, 1, 10, 100],
    'tol': [1e-3,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]  # Include default value and the range
}

combinations = list(product(*param_grid.values()))
num_combinations = len(combinations)
print("Number of Combinations in Grid Search:", num_combinations)
# Create an SVM classifier
svm_model = svm.SVC(probability=True, random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(svm_model, param_grid, cv=10, scoring='roc_auc')
grid_search.fit(X_train, y_train)
best_model_SVM= grid_search.best_estimator_
print('After grid_search the best model SVM is:'+str(best_model_SVM))
print(f"SVM After grid_search - AUC-ROC Score on train set: {roc_auc_score(y_train, best_model_SVM.predict_proba(X_train)[:, 1]):.4}")
print(confusion_matrix(y_true=y_train, y_pred=best_model_SVM.predict(X_train)))
print(f"SVM After grid_search - AUC-ROC Score on test set: {roc_auc_score(y_test, best_model_SVM.predict_proba(X_test)[:, 1]):.4}")
print(confusion_matrix(y_true=y_test, y_pred=best_model_SVM.predict(X_test)))


coef = best_model_SVM.coef_[0]
intercept = best_model_SVM.intercept_

# Print the equation of the decision boundary line
feature_names = X_train.columns.tolist()
print("Equation of the decision boundary line after Grid search:")
equation = ""
for feature, coefficient in zip(feature_names, coef):
    equation += f"{coefficient:.4f} * {feature} + "
equation += f"{intercept[0]:.4f} = 0"

print(equation)


############################################ Logistic Regression ############################################

X = dataForRegression.drop('DEATH_EVENT', axis=1)
y = dataForRegression['DEATH_EVENT']
# Split the data into training and testing sets (80% train, 20% test)
x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Export data train and test for regression in R - already done
# train_data = pd.concat([x_train, y_train], axis=1)
# train_data.to_csv('train_data.csv', index=False)
# test_data = pd.concat([X_test, y_test], axis=1)
# test_data.to_csv('test_data.csv', index=False)

### Rest of the code for Logistic Regression in  R ##
