import re
import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import time
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer



# CSV file import
dataset = pd.read_csv(r'C:/Users/44740/Desktop/Year 3/FYP/Python CODE/blacklist.csv')
# 26,000 blacklist urls labled 0 and 26,000 whitelist urls labled 1 so far on the CSV file (Need balanced dataset!)

# Strip trailing whitespaces in column names in the dataset
dataset.columns = dataset.columns.str.strip()

print(dataset.head())

# Data preprocessing 
def preprocess_url(url):
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Extract domain and path
    domain = parsed_url.netloc
    path = parsed_url.path

    # Extract selected lexical features
    url_length = len(url)
    domain_length = len(domain)
    path_length = len(path)
    subdomain_count = domain.count('.')
    digit_count = sum(1 for char in url if char.isdigit())
    repeated_char_count = sum(1 for char, count in Counter(url).items() if count > 1)
    uses_https = int(parsed_url.scheme == 'https')
    special_char_count = sum(1 for char in url if char not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    ip_address_count = int(bool(re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', url)))
    uppercase_lower_ratio = sum(1 for char in url if char.isupper()) / max(1, sum(1 for char in url if char.islower()))

    # Store dictionary of features
    processed_features = {
        'url_length': url_length,
        'domain_length': domain_length,
        'path_length': path_length,
        'subdomain_count': subdomain_count,
        'digit_count': digit_count,
        'repeated_char_count': repeated_char_count,
        'uses_https': uses_https,
        'special_char_count': special_char_count,
        'ip_address_count': ip_address_count,
        'uppercase_lower_ratio': uppercase_lower_ratio
    }
    
    return processed_features
# Apply preprocessing to dataset
dataset['processed_features'] = dataset['URL'].apply(preprocess_url)
print(dataset.head())

# Convert processed features into DataFrame
processed_features = pd.DataFrame(dataset['processed_features'].tolist())

# Rename columns
processed_features.columns = ['url_length', 'domain_length', 'path_length', 'subdomain_count', 'digit_count', 'repeated_char_count', 'uses_https', 'special_char_count', 'ip_address_count', 'uppercase_lower_ratio']

# Concatenate processed features with the original dataset
dataset = pd.concat([dataset, processed_features], axis=1)

# Split the dataset into features (X) and labels (Y)
X = dataset[['url_length', 'domain_length', 'path_length', 'subdomain_count', 'digit_count', 'repeated_char_count', 'uses_https', 'special_char_count', 'ip_address_count', 'uppercase_lower_ratio']]
Y = dataset['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2) 

# Define preprocessing transformers
extracted_features = StandardScaler() 

# Combine preprocessing steps using ColumnTransformer for training data
preprocessor = ColumnTransformer(
    transformers=[
        ('lexical', extracted_features, ['url_length', 'domain_length', 'path_length', 'subdomain_count', 'digit_count', 'repeated_char_count', 'uses_https', 'special_char_count', 'ip_address_count', 'uppercase_lower_ratio'])
    ])

# Define a simple machine learning pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
   ('classifier', RandomForestClassifier( class_weight='balanced', random_state=3))
    #('classifier', XGBClassifier())
    #('classifier', SVC())  
    #('classifier', ExtraTreesClassifier())
])


# Record the start time for classifier training
start_time = time.time()

# Train the model
model_pipeline.fit(X_train, y_train)

# Record the end time for classifier training
end_time = time.time()

# Calculateing the run time of the model
run_time = end_time - start_time

print(f"Run time for classifier training: {run_time} seconds")

# Prediction using the model
prediction = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
print(f"Accuracy: {accuracy}")

classification_report_result = classification_report(y_test, prediction)
print("Classification Report:\n", classification_report_result)
#------------------------------------------------------
# Print confusion matrix
confusion = confusion_matrix(y_test, prediction)
print("Confusion Matrix:\n", confusion)

# Print or visualize feature importance
if isinstance(model_pipeline.named_steps['classifier'], RandomForestClassifier):
    feature_importance = model_pipeline.named_steps['classifier'].feature_importances_

    # Print top features
    print("Top Features:")
    for feature, importance in zip(X.columns, feature_importance):
        print(f"{feature}: {importance}")

#------------------------------------------------------------------
# Ask user to enter a URL for prediction
user_url = input("Enter the URL to check: ")

# Preprocess the user input URL
user_processed_url = preprocess_url(user_url)

# Convert the processed URL features into a DataFrame
user_input = pd.DataFrame([user_processed_url])

# Make predictions
prediction = model_pipeline.predict(user_input)

# Take the first prediction (assuming it's a single sample)
outcome = prediction[0]

# Print the outcome
if (outcome == 0):
    print ('Malicious URL')
else: 
    print('Safe URL')

print("_______________________________________________________")

def predict_url(url):
   
    # Preprocess the URL
    processed_url = preprocess_url(url)

    # Convert processed URL features into a DataFrame
    user_input = pd.DataFrame([processed_url])

    # Extract features
    #features = [processed_url[feature] for feature in processed_url]

    # Check if the user input URL is in the dataset
    if processed_url in dataset['processed_features'].values:
        # Find the corresponding row in the dataset
        matching_row = dataset[dataset['processed_features'].eq(processed_url)].iloc[0]
        # Extract the label from the matching row
        label = matching_row['Label']
        return 'malicious in dataset' if label == 0 else 'safe in dataset'
    else:
        # Proceed with model prediction
        prediction = model_pipeline.predict(user_input)[0]
        return 'Malicious' if prediction == 0 else 'Safe'
