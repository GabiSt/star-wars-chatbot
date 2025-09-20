import sys
import pandas as pd
import numpy as np
import re
import random
from typing import List, Dict, Any
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import pygame  # Added for audio playback

# Force CPU usage to avoid CUDA issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize pygame mixer
pygame.mixer.init()

# PyQt5 imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QLineEdit, QPushButton, 
                             QLabel, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor

class StarWarsChatbot:
    def __init__(self):
        tf.keras.backend.clear_session()

        # Load data
        self.df = pd.read_csv('star_wars_dataset.csv', usecols=['intent', 'pattern', 'response'])
        self.df.columns = ['intent', 'pattern', 'response']  # Ensure correct column names
        self.df = self.df.dropna()
        self.df = self.df[self.df['pattern'].str.strip() != '']

        # Try to load pre-trained model and tokenizers
        try:
            self.model = tf.keras.models.load_model('star_wars_rnn_model.h5')
            with open('tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            with open('label_encoder.pickle', 'rb') as handle:
                self.le = pickle.load(handle)
            self.max_length = self.model.input_shape[1]
            print("✅ Successfully loaded pre-trained LSTM model and tokenizers.")

        except Exception as e:
            print(f"❌ Failed to load pre-trained model: {e}")
            print("Training a new LSTM model...")
            self.train_new_model()

    def train_new_model(self):
        patterns = self.df['pattern'].astype(str).tolist()
        intents = self.df['intent'].astype(str).tolist()

        # Preprocess the patterns
        processed_patterns = [self.preprocess_text(p) for p in patterns]

        # Tokenization
        self.tokenizer = Tokenizer(num_words=500, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(processed_patterns)
        sequences = self.tokenizer.texts_to_sequences(processed_patterns)
        self.max_length = max(len(seq) for seq in sequences) if sequences else 10

        # Label encoding
        self.le = LabelEncoder()
        labels = self.le.fit_transform(intents)

        # Build and train LSTM model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                input_dim=len(self.tokenizer.word_index) + 1,
                output_dim=128,
                input_length=self.max_length
            ),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.le.classes_), activation='softmax')
        ])

        self.model.compile(
            loss='sparse_categorical_crossentropy', 
            optimizer='adam', 
            metrics=['accuracy']
        )

        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        self.model.fit(
            padded_sequences, 
            labels, 
            epochs=50, 
            batch_size=32,
            verbose=1,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ]
        )

        # Save the new model and tokenizers
        self.model.save('star_wars_rnn_model.h5')
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('label_encoder.pickle', 'wb') as handle:
            pickle.dump(self.le, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("✅ New LSTM model trained and saved successfully.")

    def preprocess_text(self, text):
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def get_response(self, text):
        processed_text = self.preprocess_text(text)

        if not processed_text:
            return "Please ask me something about Star Wars!"

        try:
            seq = self.tokenizer.texts_to_sequences([processed_text])

            if not seq or len(seq[0]) == 0:
                return self.rule_based_fallback(text)

            padded = pad_sequences(seq, maxlen=self.max_length, padding='post')
            pred = self.model.predict(padded, verbose=0)

            intent_index = np.argmax(pred)
            confidence = np.max(pred)

            confidence_threshold = 0.6
            if any(word in processed_text for word in ['hello', 'hi', 'hey', 'bye', 'goodbye']):
                confidence_threshold = 0.3

            if confidence < confidence_threshold:
                return self.rule_based_fallback(text)

            intent_tag = self.le.inverse_transform([intent_index])[0]

            # 🔹 Check if exact pattern exists
            match = self.df[(self.df['intent'] == intent_tag) & 
                            (self.df['pattern'].str.lower() == processed_text.lower())]

            if not match.empty:
                return match['response'].iloc[0]

            # fallback: random response from intent
            resp_list = self.df[self.df['intent'] == intent_tag]['response'].tolist()
            return np.random.choice(resp_list) if resp_list else "I don't have a response for that."

        except Exception as e:
            print(f"Error in model prediction: {e}")
            return self.rule_based_fallback(text)

    def rule_based_fallback(self, text):
        text_lower = text.lower().strip()

        keyword_responses = {
            'luke': "Luke Skywalker is a Jedi Knight and the son of Anakin Skywalker.",
            'vader': "Darth Vader is a Sith Lord and former Jedi Knight Anakin Skywalker.",
            'anakin': "Anakin Skywalker was a Jedi Knight who became Darth Vader.",
            'force': "The Force is a metaphysical power that Jedi and Sith use.",
            'lightsaber': "A lightsaber is an energy sword used by Jedi and Sith.",
            'falcon': "The Millennium Falcon is a fast spaceship piloted by Han Solo.",
            'jedi': "Jedi are peacekeepers who use the Force for good.",
            'sith': "Sith are dark side users who seek power and control.",
            'leia': "Princess Leia is a leader of the Rebel Alliance.",
            'han': "Han Solo is a smuggler who becomes a hero of the Rebellion.",
            'yoda': "Yoda is a wise and powerful Jedi Master.",
            'star wars': "Star Wars is an epic space opera franchise created by George Lucas!",
            'skywalker': "The Skywalker family is central to the Star Wars saga.",
            'death star': "The Death Star is a massive space station and superweapon.",
            'rebel': "The Rebel Alliance fights against the Galactic Empire.",
            'empire': "The Galactic Empire is the main antagonist faction in the original trilogy.",
            'droid': "Droids are robotic beings that serve various functions in the Star Wars universe.",
            'clone': "Clone troopers were soldiers created from the genetic template of Jango Fett.",
            'wookiee': "Wookiees are a species of tall, hairy humanoids from the planet Kashyyyk.",
            'ewok': "Ewoks are small, furry creatures native to the forest moon of Endor."
        }

        for keyword, response in keyword_responses.items():
            if keyword in text_lower:
                return response

        fallback_responses = [
            "I'm not sure about that. Ask me something about Star Wars!",
            "That's an interesting question! Try asking about characters, ships, or the Force.",
            "The Force is not strong with that question. Try asking about Star Wars!",
            "I'm still learning about the Star Wars universe. Could you ask something else?",
            "May the Force be with you! Try asking about Jedi, Sith, or characters from the movies."
        ]

        return random.choice(fallback_responses)

class ChatWorker(QThread):
    response_ready = pyqtSignal(str)

    def __init__(self, chatbot, message):
        super().__init__()
        self.chatbot = chatbot
        self.message = message

    def run(self):
        response = self.chatbot.get_response(self.message)
        self.response_ready.emit(response)

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.chatbot = StarWarsChatbot()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Star Wars Chatbot (LSTM)')
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        title = QLabel('🌟 Star Wars Chatbot 🌟')
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont('Arial', 18, QFont.Bold)
        title.setFont(title_font)
        title.setStyleSheet("""
            QLabel {
                color: #FFD700;
                background-color: #000064;
                padding: 15px;
                border: 2px solid #FFD700;
                border-radius: 10px;
                margin: 5px;
            }
        """)
        layout.addWidget(title)

        chat_frame = QFrame()
        chat_frame.setStyleSheet("""
            QFrame {
                background-color: #000032;
                border: 2px solid #FFD700;
                border-radius: 10px;
                margin: 5px;
            }
        """)
        chat_layout = QVBoxLayout(chat_frame)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #000020;
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
                padding: 15px;
                font-family: Arial;
                font-size: 13px;
                margin: 5px;
            }
        """)
        chat_layout.addWidget(self.chat_display)

        layout.addWidget(chat_frame)

        input_frame = QFrame()
        input_frame.setStyleSheet("""
            QFrame {
                background-color: #000032;
                border: 2px solid #FFD700;
                border-radius: 10px;
                margin: 5px;
            }
        """)
        input_layout = QHBoxLayout(input_frame)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask me about Star Wars... (Press Enter to send)")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: #000040;
                color: #FFFFFF;
                border: 2px solid #FFD700;
                border-radius: 8px;
                padding: 12px;
                font-family: Arial;
                font-size: 13px;
                margin: 5px;
            }
            QLineEdit:focus {
                border: 2px solid #FFEE58;
            }
        """)
        self.input_field.returnPressed.connect(self.send_message)

        self.send_button = QPushButton('🚀 Send')
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #FFD700;
                color: #000064;
                border: 2px solid #FFD700;
                border-radius: 8px;
                padding: 12px;
                font-family: Arial;
                font-size: 13px;
                font-weight: bold;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #FFEE58;
                border-color: #FFEE58;
            }
            QPushButton:pressed {
                background-color: #FFC400;
                border-color: #FFC400;
            }
        """)
        self.send_button.clicked.connect(self.send_message)

        input_layout.addWidget(self.input_field, 4)
        input_layout.addWidget(self.send_button, 1)

        layout.addWidget(input_frame)

        self.status_label = QLabel("🤖 LSTM Model Loaded - Ready to chat!")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #00FF00;
                padding: 8px;
                background-color: #002000;
                border-radius: 5px;
                margin: 5px;
            }
        """)
        layout.addWidget(self.status_label)

        self.add_message("Chatbot", "Hello! I'm a Star Wars chatbot powered by LSTM! 🌌")

    def add_message(self, sender, message):
        timestamp = pd.Timestamp.now().strftime("%H:%M")

        if sender == "You":
            html = f'''
            <div style="margin: 10px; text-align: right;">
                <div style="color: #FFD700; font-weight: bold; font-size: 11px;">You • {timestamp}</div>
                <div style="background-color: #1E3A8A; color: white; padding: 10px; border-radius: 15px; 
                          border-top-right-radius: 5px; display: inline-block; max-width: 70%; 
                          margin: 5px; text-align: left;">
                    {message}
                </div>
            </div>
            '''
        else:
            html = f'''
            <div style="margin: 10px; text-align: left;">
                <div style="color: #00FF00; font-weight: bold; font-size: 11px;">Chatbot • {timestamp}</div>
                <div style="background-color: #2D5016; color: white; padding: 10px; border-radius: 15px; 
                          border-top-left-radius: 5px; display: inline-block; max-width: 70%; 
                          margin: 5px; text-align: left;">
                    {message}
                </div>
            </div>
            '''

        self.chat_display.append(html)
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

        if sender == "Chatbot":
            if "General Kenobi!?" in message:
                self.play_audio("sfx/general-kenobi.mp3")
            elif "Do. Or do not. There is no try." in message:
                self.play_audio("sfx/yoda-do-or-do-not.mp3")
            elif "Hello, there!" in message:
                self.play_audio("sfx/hello-there.mp3")
            elif "I find your lack of faith disturbing." in message:
                self.play_audio("sfx/lack-of-faith.mp3")
            elif "NOOO!" in message:
                self.play_audio("sfx/nooo.mp3")
            elif "DO IT!" in message:
                self.play_audio("sfx/do-it.mp3")

    def play_audio(self, path):
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Could not play audio: {e}")

    def send_message(self):
        message = self.input_field.text().strip()
        if not message:
            return

        self.add_message("You", message)
        self.input_field.clear()

        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        self.status_label.setText("⏳ Processing your question...")

        self.worker = ChatWorker(self.chatbot, message)
        self.worker.response_ready.connect(self.handle_response)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def handle_response(self, response):
        self.add_message("Chatbot", response)
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.status_label.setText("🤖 LSTM Model Loaded - Ready to chat!")
        self.input_field.setFocus()


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(0, 0, 30))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(0, 0, 20))
    palette.setColor(QPalette.AlternateBase, QColor(0, 0, 40))
    palette.setColor(QPalette.Text, Qt.white)
    app.setPalette(palette)
    
    # Create and show the main window
    window = ChatWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
