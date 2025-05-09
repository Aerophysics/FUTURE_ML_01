import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score
from sklearn.naive_bayes import MultinomialNB
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import warnings
warnings.filterwarnings('ignore')

# Download dataset from Kaggle
path = kagglehub.dataset_download("waseemalastal/customer-support-ticket-dataset")
df = pd.read_csv(f"{path}/customer_support_tickets.csv", encoding='latin1')

# Display DataFrame information
print("\nDataFrame Info:")
print(df.info())
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Data Preprocessing
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = ''.join([c for c in text if c.isalnum() or c.isspace()])
        return text
    return ""

# Process the text data
df['processed_text'] = df['Ticket Description'].apply(preprocess_text)
y = df['Ticket Type']

# Create TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000, min_df=2)
X_tfidf = vectorizer.fit_transform(df['processed_text'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Make predictions
y_pred_nb = nb_classifier.predict(X_test)

# Calculate metrics for Naive Bayes
nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_precision = precision_score(y_test, y_pred_nb, average='weighted')
nb_cm = confusion_matrix(y_test, y_pred_nb)

print("\nNaive Bayes Model Performance:")
print(f"Accuracy: {nb_accuracy:.4f}")
print(f"Precision: {nb_precision:.4f}")

# Plot confusion matrix for Naive Bayes
plt.figure(figsize=(10, 8))
sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Naive Bayes Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('nb_confusion_matrix.png')
plt.close()

# Deep Learning Model
# Prepare data for LSTM
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['processed_text'])
sequences = tokenizer.texts_to_sequences(df['processed_text'])
X_lstm = pad_sequences(sequences, maxlen=100)

# Convert labels to numerical values
label_map = {label: idx for idx, label in enumerate(df['Ticket Type'].unique())}
y_lstm = df['Ticket Type'].map(label_map)

# Split data for LSTM
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    X_lstm, y_lstm, test_size=0.2, random_state=42
)

# Build LSTM model
lstm_model = Sequential([
    Embedding(5000, 100, input_length=100),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(len(label_map), activation='softmax')
])

lstm_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train LSTM model
history = lstm_model.fit(
    X_train_lstm, y_train_lstm,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Make predictions with LSTM
y_pred_lstm = np.argmax(lstm_model.predict(X_test_lstm), axis=1)

# Calculate metrics for LSTM
lstm_accuracy = accuracy_score(y_test_lstm, y_pred_lstm)
lstm_precision = precision_score(y_test_lstm, y_pred_lstm, average='weighted')
lstm_cm = confusion_matrix(y_test_lstm, y_pred_lstm)

print("\nLSTM Model Performance:")
print(f"Accuracy: {lstm_accuracy:.4f}")
print(f"Precision: {lstm_precision:.4f}")

# Plot confusion matrix for LSTM
plt.figure(figsize=(10, 8))
sns.heatmap(lstm_cm, annot=True, fmt='d', cmap='Blues')
plt.title('LSTM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('lstm_confusion_matrix.png')
plt.close()

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('LSTM Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('lstm_training_history.png')
plt.close()

# Print detailed classification reports
print("\nNaive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))

print("\nLSTM Classification Report:")
print(classification_report(y_test_lstm, y_pred_lstm))

# Save predictions
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted_NB': y_pred_nb,
    'Predicted_LSTM': y_pred_lstm
})
predictions_df.to_csv('chatbot_predictions.csv', index=False)

print("\nResults have been saved to:")
print("- nb_confusion_matrix.png")
print("- lstm_confusion_matrix.png")
print("- lstm_training_history.png")
print("- chatbot_predictions.csv")