import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor



# Load your dataset
data = pd.read_csv(r"C:\Users\merie\OneDrive\Bureau\WGU\D600\Task2-D600\D600 Task 2 Dataset 1 Housing Information.csv")

# Check unique values in 'Garage'
print(data['Garage'].unique())

# Convert the variable 'Garage' to binary (Yes: 1, No: 0)
data['Garage'] = data['Garage'].map({'Yes': 1, 'No': 0})

## Descriptive statistics
# Dependent Variable: Garage
garage_counts = data['Garage'].value_counts()
garage_mode = data['Garage'].mode()

# Independent Variable: Home Square Footage
square_footage_stats = data['SquareFootage'].describe()
square_footage_range = data['SquareFootage'].max() - data['SquareFootage'].min()

# Independent Variable: Backyard Space
backyard_space_stats = data['BackyardSpace'].describe()
backyard_space_range = data['BackyardSpace'].max() - data['BackyardSpace'].min()

# Print results
print("Garage Counts:\n", garage_counts)
print("Garage Mode:", garage_mode)

print("\nHome Square Footage Statistics:")
print("Count:", square_footage_stats['count'])
print("Mean:", square_footage_stats['mean'])
print("Min:", square_footage_stats['min'])
print("Max:", square_footage_stats['max'])
print("Range:", square_footage_range)

print("\nBackyard Space Statistics:")
print("Count:", backyard_space_stats['count'])
print("Mean:", backyard_space_stats['mean'])
print("Min:", backyard_space_stats['min'])
print("Max:", backyard_space_stats['max'])
print("Range:", backyard_space_range)

## Univariate visualization 
# Home Square Footage
plt.figure(figsize=(10, 5))
sns.histplot(data['SquareFootage'], bins=30, kde=True)
plt.title('Distribution of Home Square Footage')
plt.xlabel('Square Footage')
plt.ylabel('Frequency')
plt.show()

# Backyard Space
plt.figure(figsize=(10, 5))
sns.histplot(data['BackyardSpace'], bins=30, kde=True)
plt.title('Distribution of Backyard Space')
plt.xlabel('Backyard Space')
plt.ylabel('Frequency')
plt.show()

# Houses with/without Garage
plt.figure(figsize=(6, 4))
sns.countplot(x='Garage', data=data)
plt.title('Count of Houses with/without Garage')
plt.xlabel('Garage')
plt.ylabel('Count')
plt.show()

# Bivariate visualization
# Square Footage by Garage
plt.figure(figsize=(8, 5))
sns.boxplot(x='Garage', y='SquareFootage', data=data)
plt.title('Square Footage by Garage Availability')
plt.xlabel('Garage')
plt.ylabel('Square Footage')
plt.show()

# Backyard Space by Garage
plt.figure(figsize=(8, 5))
sns.boxplot(x='Garage', y='BackyardSpace', data=data)
plt.title('Backyard Space by Garage Availability')
plt.xlabel('Garage')
plt.ylabel('Backyard Space')
plt.show()

# Square Footage vs. Backyard Space Colored
plt.figure(figsize=(10, 5))
sns.scatterplot(x='SquareFootage', y='BackyardSpace', hue='Garage', style='Garage', data=data)
plt.title('Square Footage vs. Backyard Space Colored by Garage Availability')
plt.xlabel('Square Footage')
plt.ylabel('Backyard Space')
plt.legend(title='Garage')
plt.show()

# Prepare the data for modeling
selected_data = data[['Garage', 'SquareFootage', 'BackyardSpace']]
X = selected_data[['SquareFootage', 'BackyardSpace']]
y = selected_data['Garage']  

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the datasets to CSV files
train_data = pd.DataFrame(X_train)
train_data['Garage'] = y_train
test_data = pd.DataFrame(X_test)
test_data['Garage'] = y_test

train_data.to_csv(r"C:\Users\merie\OneDrive\Bureau\WGU\D600\Task2-D600\Training_Dataset.csv", index=False)
test_data.to_csv(r"C:\Users\merie\OneDrive\Bureau\WGU\D600\Task2-D600\Testing_Dataset.csv", index=False)

print("Datasets created and saved successfully.")

# Apply SMOTE in case the dataset is imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Add a constant (intercept) / No scaling
X_train_sm = sm.add_constant(X_train_resampled)
X_test_sm = sm.add_constant(X_test)

# Backward Elimination for Logistic Regression
def backward_elimination_logit(X, y, significance_level=0.05):
    # make copy to avoid modifying the original data
    X = X.copy()  
    while True:
        # Fit the logistic regression model
        model = sm.Logit(y, X).fit()
        # Get the p-values 
        p_values = model.pvalues  

        print(f"Current model p-values:\n{p_values}\n")
        
        # Find the maximum p-value in the model
        max_p_value = p_values.max()
        
        # If the maximum p-value is greater than the significance level, drop the feature with the highest p-value
        if max_p_value > significance_level:
            # Get the feature with the highest p-value
            excluded_feature = p_values.idxmax()  
            print(f"Dropping feature: {excluded_feature} with p-value: {max_p_value}")
            # Drop the feature with the highest p-value
            X = X.drop(columns=[excluded_feature])  
        else:
            # If all features are statistically significant, exit 
            break  

    return model

# Apply backward elimination on the training set
optimized_model = backward_elimination_logit(X_train_sm, y_train_resampled)

# Print the summary of the optimized logistic regression model
print("\nOptimized Logistic Regression Model Summary:")
print(optimized_model.summary())

# Extract model parameters
aic = optimized_model.aic
bic = optimized_model.bic
llf = optimized_model.llf
llnull = optimized_model.llnull
pseudo_r2 = 1 - (llf / llnull)
coefficients = optimized_model.params
p_values = optimized_model.pvalues

# Print extracted parameters
print(f"\nAIC: {aic}")
print(f"BIC: {bic}")
print(f"Pseudo R2: {pseudo_r2}")
print("Coefficient Estimates:\n", coefficients)
print("P-values:\n", p_values)

# Generate confusion matrix and calculate accuracy for the training set
# Make sure the features match the model's expectations
# Get the selected feature names
selected_features = optimized_model.model.exog_names 
# Use only the selected features for prediction 
X_train_selected = X_train_sm[selected_features]  

# Predict on the training set using the optimized model
y_train_pred_prob = optimized_model.predict(X_train_selected)

# Convert the predicted probabilities to binary values (0 or 1)
y_train_pred_binary = (y_train_pred_prob >= 0.5).astype(int)

# Confusion matrix and accuracy
conf_matrix = confusion_matrix(y_train_resampled, y_train_pred_binary)
accuracy = accuracy_score(y_train_resampled, y_train_pred_binary)

# Print the confusion matrix and accuracy
print("\nConfusion Matrix (Training Set):")
print(conf_matrix)
print(f"Accuracy (Training Set): {accuracy:.2f}")

# Evaluate the performance of the prediction model on the test data based on the confusion matrix and accuracy
# Use only the selected features from the test set
X_test_selected = X_test_sm[selected_features]

# Predict on the test set using the optimized model
y_test_pred_prob = optimized_model.predict(X_test_selected)

# Convert the predicted probabilities to binary values (0 or 1) based on a threshold of 0.5
y_test_pred_binary = (y_test_pred_prob >= 0.5).astype(int)

# Confusion matrix and accuracy
conf_matrix_test = confusion_matrix(y_test, y_test_pred_binary)
accuracy_test = accuracy_score(y_test, y_test_pred_binary)

# Print the confusion matrix and accuracy for the test set
print("\nConfusion Matrix (Test Set):")
print(conf_matrix_test)
print(f"Accuracy (Test Set): {accuracy_test:.2f}")

# Check the 4 assumptions of logistic regression

# 1. Linearity of the Logit by using scotter plot
# Standardize the features 
scaler = StandardScaler()
X_train_resampled_scaled = pd.DataFrame(scaler.fit_transform(X_train_resampled), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Add the constant (intercept) to the training and testing data
X_train_with_constant = sm.add_constant(X_train_resampled_scaled)
X_test_with_constant = sm.add_constant(X_test_scaled)

# Fit the logistic regression model
optimized_model = sm.Logit(y_train_resampled, X_train_with_constant).fit()

# Predict the probabilities 
y_train_pred_prob = optimized_model.predict(X_train_with_constant)

# Calculate the logit (log-odds)
logit = np.log(y_train_pred_prob / (1 - y_train_pred_prob))

# Scatter plot of SquareFootage vs. Logit (log-odds)
plt.figure(figsize=(8, 5))
plt.scatter(X_train_with_constant['SquareFootage'], logit, alpha=0.5)
plt.title('SquareFootage vs. Logit (log-odds)')
plt.xlabel('SquareFootage')
plt.ylabel('Logit (log-odds)')
plt.grid(True)
plt.show()

# Scatter plot of BackyardSpace vs. Logit (log-odds)
plt.figure(figsize=(8, 5))
plt.scatter(X_train_with_constant['BackyardSpace'], logit, alpha=0.5)
plt.title('BackyardSpace vs. Logit (log-odds)')
plt.xlabel('BackyardSpace')
plt.ylabel('Logit (log-odds)')
plt.grid(True)
plt.show()

# 2.Independence of Observations (Durbin-Watson test)
residuals = optimized_model.resid_response  # residuals from the optimized logistic model
dw_statistic = durbin_watson(residuals)
print(f"Durbin-Watson Statistic: {dw_statistic:.2f}")
if dw_statistic < 1.5 or dw_statistic > 2.5:
    print("The residuals may be autocorrelated")

# 3. No Perfect Correlation between Independent Variables 
# Compute VIF (Variance inflation factor) for each feature
vif_data = pd.DataFrame()
vif_data["feature"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
print("\nVariance Inflation Factor (VIF):\n", vif_data)

# Check if any VIF is greater than 10 (which indicates a multicollinearity problem)
for feature, vif in zip(vif_data["feature"], vif_data["VIF"]):
    if vif > 10:
        print(f"High multicollinearity detected for feature '{feature}' with VIF = {vif:.2f}")

# 4. Binary Outcome 
unique_values = y_train.unique()
print(f"Unique values of the dependent variable: {unique_values}")
if len(unique_values) != 2:
    print("The dependent variable is not binary")

