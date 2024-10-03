import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import nltk
from spellchecker import SpellChecker 
import string
import time
import re
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# CSV file import
textdataset = pd.read_csv(r'C:/Users/44740/Desktop/Year 3/FYP/Python CODE/spam.csv')

# Initializing SpellChecker
spell = SpellChecker()

def count_misspelled_words(text):
    # Tokenize the text into words
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and punctuation
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word.lower() not in string.punctuation]
    
    # Count the number of misspelled words
    misspelled_count = sum(1 for word in tokens if word not in spell)
    
    return misspelled_count


# Clean data (Removing stop words, punctuation and links)
def preprocess_text(text):
    # Remove links
    text = re.sub(r'http\S+', '', text)
    
    # Tokenize the text into words
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and punctuation
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word.lower() not in string.punctuation]
    
    # Join the tokens back into a single string
    clean_text = ' '.join(tokens)
    
    # Count misspelled words
    misspelled_count = count_misspelled_words(clean_text)
    
    # Count presence of urgent phrases
    urgent_words = [
    "act now", "apply now", "become a member", "call now", "click below",
    "click here", "get it now", "do it today", "dont delete", "exclusive deal",
    "get started now", "important information regarding", "information you requested",
    "instant", "limited time", "new customers only", "order now", "please read",
    "see for yourself", "sign up free", "take action", "this wont last", "urgent",
    "what are you waiting for?", "while supplies last", "will not believe your eyes",
    "winner", "winning", "you are a winner", "you have been selected"
    ]

    urgent_count = sum(1 for word in urgent_words if word in clean_text)


    # Count the occurrence of special characters
    special_char_count = sum(1 for char in clean_text if char not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    
    # Calculate text length
    text_length = len(text)

    digit_count = sum(1 for char in clean_text if char.isdigit())

    
    return clean_text, special_char_count, text_length, misspelled_count, urgent_count, digit_count

# Apply preprocessing to the dataset
textdataset['clean_text'], textdataset['special_char_count'], textdataset['text_length'], textdataset['misspelled_count'], textdataset['urgent_count'], textdataset ['digit_count'] = zip(*textdataset['Text'].apply(preprocess_text))

# Display the preprocessed dataset
print(textdataset.head())


# Convert string labels to numerical values
textdataset['Label'] = textdataset['Label'].map({'ham': 0, 'spam': 1})

# Split the dataset into features (X) and labels (Y)
X = textdataset[['clean_text', 'special_char_count', 'text_length', 'misspelled_count',  'urgent_count', 'digit_count']]
Y = textdataset['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Define preprocessing transformers
text_transformer = TfidfVectorizer()
extracted_features = StandardScaler()

# Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, 'clean_text'),
        ('lexical', extracted_features, ['special_char_count', 'text_length', 'misspelled_count', 'urgent_count', 'digit_count'])
    ])

# Define a simple machine learning pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier( class_weight='balanced', random_state=3))
    #('classifier', XGBClassifier())
   # ('classifier', SVC())  
   # ('classifier', ExtraTreesClassifier())
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
classification_report_output = classification_report(y_test, prediction)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report_output)

# Print confusion matrix
confusion = confusion_matrix(y_test, prediction)
print("Confusion Matrix:\n", confusion)

# Print or visualize feature importance
if isinstance(model_pipeline.named_steps['classifier'], RandomForestClassifier):
    feature_importance = model_pipeline.named_steps['classifier'].feature_importances_

    # Print top features
    print("Top Features:")
    for feature, importance in zip(X_test.columns, feature_importance):
        print(f"{feature}: {importance}")

# Take user input
user_input_text = input("Enter the text to classify: ")

# Check if user input is in the dataset
if user_input_text in textdataset['Text'].values:
    label = textdataset.loc[textdataset['Text'] == user_input_text, 'Label'].iloc[0]
    if label == 'spam':
        print("The text is labeled as malicious.")
    else:
        print("The text is labeled as safe.")
else:
    # Preprocess the user input
    clean_text, special_char_count, text_length, misspelled_count, urgent_count, digit_count = preprocess_text(user_input_text)

    # Prepare input for prediction
    user_input_features = pd.DataFrame({
        'clean_text': [clean_text],
        'special_char_count': [special_char_count],
        'text_length': [text_length],
        'misspelled_count': [misspelled_count],
        'urgent_count': [urgent_count],
        'digit_count' : [digit_count]
    })

    # Make prediction
    prediction = model_pipeline.predict(user_input_features)

    # Display prediction
    if prediction[0] == 0:
        print("The text is predicted to be malicious.")
    else:
        print("The text is predicted to be benign.")

def predict_text(text):

    # Check if the user input text is in the dataset
    if text in textdataset['Text'].values:
        # Get the corresponding label from the dataset
        label = textdataset.loc[textdataset['Text'] == text, 'Label'].iloc[0]
        if label == 'spam':
            return 'Malicious as per the dataset'  # If the label is spam, return malicious
        else:
            return 'Safe as per the dataset dataset'  # If the label is not spam, return safe
    else:
        clean_text, special_char_count, text_length, misspelled_count, urgent_count, digit_count = preprocess_text(text)
        # Prepare input features for prediction
        user_input_features = pd.DataFrame({
            'clean_text': [clean_text],
            'special_char_count': [special_char_count],
            'text_length': [text_length],
            'misspelled_count': [misspelled_count],
            'urgent_count': [urgent_count],
            'digit_count' : [digit_count]
        })
        # Make prediction using the model
        prediction = model_pipeline.predict(user_input_features)
        return 'Malicious' if prediction[0] == 0 else 'Safe'

