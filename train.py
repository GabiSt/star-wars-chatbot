import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import re
import tensorflow as tf

def augment_dataset(df):
    """Augment the dataset with synonym replacements"""
    augmented_data = []
    
    synonym_map = {
        'who is': ['tell me about', 'what do you know about', 'information about', 
                  'can you tell me about', 'who was'],
        'what is': ['tell me about', 'what do you know about', 'information about',
                   'can you explain', 'describe'],
        'hello': ['hi', 'hey', 'greetings', 'howdy'],
        'bye': ['goodbye', 'see you', 'farewell', 'later'],
        'jedi': ['jedi knight', 'force user', 'lightsaber wielder'],
        'sith': ['sith lord', 'dark side user', 'dark side practitioner']
    }
    
    for _, row in df.iterrows():
        augmented_data.append(row)
        pattern = row['pattern'].lower()
        
        # Add variations with synonyms
        for original, replacements in synonym_map.items():
            if original in pattern:
                for replacement in replacements:
                    new_pattern = pattern.replace(original, replacement)
                    if new_pattern != pattern:
                        new_row = row.copy()
                        new_row['pattern'] = new_pattern.capitalize()
                        augmented_data.append(new_row)
    
    return pd.DataFrame(augmented_data)

# Load and augment dataset
print("Loading and augmenting dataset...")
df = pd.read_csv('star_wars_dataset.csv')
df = df.dropna()
df = df[df['pattern'].str.strip() != '']

# Augment the dataset
df_augmented = augment_dataset(df)
print(f"Original samples: {len(df)}")
print(f"Augmented samples: {len(df_augmented)}")

def preprocess_text(text):
    """Apply consistent preprocessing"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Apply preprocessing
df_augmented['pattern_processed'] = df_augmented['pattern'].apply(preprocess_text)

print("Final Dataset Analysis:")
print(f"Total samples: {len(df_augmented)}")
print(f"Number of unique intents: {df_augmented['intent'].nunique()}")
print("\nSamples per intent:")
print(df_augmented['intent'].value_counts())

# Preprocess data
texts = df_augmented['pattern_processed'].values
labels = df_augmented['intent'].values

# Tokenize text
tokenizer = Tokenizer(num_words=500, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(texts)
max_length = max(len(seq) for seq in sequences) if sequences else 15
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(np.unique(encoded_labels))

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, encoded_labels, 
    test_size=0.2, random_state=42, stratify=encoded_labels
)

# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=128,
        input_length=max_length
    ),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train model
print("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]
)

# Save model
model.save('star_wars_rnn_model.h5')

# Save preprocessing objects
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('label_encoder.pickle', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Generate predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Create confusion matrix
class_names = label_encoder.classes_
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Print results
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, 
                           target_names=class_names, zero_division=0))

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Enhanced testing
test_phrases = [
    "Who is Luke Skywalker?",
    "Tell me about Darth Vader",
    "What is the force?",
    "Tell me about lightsabers",
    "Hello there",
    "Bye",
    "What about his father?",
    "Who is Yoda?",
    "Information about the millennium falcon",
    "What do you know about jedi?"
]

print("\nEnhanced Testing:")
for phrase in test_phrases:
    processed = preprocess_text(phrase)
    seq = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')
    pred = model.predict(padded, verbose=0)
    intent_index = np.argmax(pred)
    confidence = np.max(pred)
    intent = label_encoder.inverse_transform([intent_index])[0]
    
    print(f"'{phrase}' -> {intent} (confidence: {confidence:.2f})")
