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

class ConversationMemory:
    def __init__(self, max_history=5):
        self.max_history = max_history
        self.reset()
    
    def reset(self):
        self.history = []
        self.current_topic = None
        self.current_entity = None
    
    def add_message(self, role: str, message: str):
        self.history.append({"role": role, "message": message})
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract potential entities from text for context"""
        entities = {}
        text_lower = text.lower()
        
        # Star Wars character names
        characters = [
            'luke', 'vader', 'anakin', 'leia', 'han', 'chewbacca', 'yoda', 
            'obi-wan', 'kenobi', 'padme', 'rey', 'finn', 'poe', 'kylo', 'snoke',
            'palpatine', 'sidious', 'dooku', 'maul', 'jango', 'boba', 'jabba',
            'lando', 'wedge', 'ackbar', 'mace', 'qui-gon', 'ahsoka', 'thrawn'
        ]
        
        for char in characters:
            if char in text_lower:
                entities['character'] = char
                break
        
        # Other entities
        entity_types = {
            'ship': ['falcon', 'x-wing', 'tie fighter', 'star destroyer', 'death star', 'millennium'],
            'force': ['force', 'lightside', 'dark side', 'jedi', 'sith'],
            'planet': ['tatooine', 'coruscant', 'naboo', 'hoth', 'endor', 'dagobah', 'mustafar'],
            'faction': ['rebellion', 'empire', 'republic', 'first order', 'resistance']
        }
        
        for entity_type, keywords in entity_types.items():
            for keyword in keywords:
                if keyword in text_lower and entity_type not in entities:
                    entities[entity_type] = keyword
        
        return entities
    
    def update_context(self, user_message: str, bot_response: str):
        """Update conversation context based on the latest exchange"""
        self.add_message("user", user_message)
        self.add_message("bot", bot_response)
        
        # Extract entities from user message
        entities = self.extract_entities(user_message)
        
        # Update current topic if we found entities
        if entities:
            if 'character' in entities:
                self.current_entity = entities['character']
                self.current_topic = 'character'
            elif 'ship' in entities:
                self.current_entity = entities['ship']
                self.current_topic = 'ship'
            elif 'planet' in entities:
                self.current_entity = entities['planet']
                self.current_topic = 'planet'
            elif 'faction' in entities:
                self.current_entity = entities['faction']
                self.current_topic = 'faction'
            elif 'force' in entities:
                self.current_entity = None
                self.current_topic = 'force'
    
    def contextualize_query(self, user_message: str) -> str:
        """Add context to the user's query if it seems like a follow-up"""
        if not self.current_topic or not self.history:
            return user_message
        
        # Check if this seems like a follow-up question
        follow_up_indicators = [
            'what about', 'how about', 'tell me more', 'what else',
            'and', 'what about him', 'what about her', 'what about it',
            'who is', 'what is', 'where is', 'when did', 'why did', 'how did'
        ]
        
        message_lower = user_message.lower()
        is_follow_up = any(indicator in message_lower for indicator in follow_up_indicators)
        
        # If it's a short message or seems like a follow-up, add context
        if (len(user_message.split()) < 5 or is_follow_up) and self.current_entity:
            contextualized = f"{self.current_entity} {user_message}"
            return contextualized
        
        return user_message

class StarWarsChatbot:
    def __init__(self):
        tf.keras.backend.clear_session()
        
        # Initialize conversation memory
        self.memory = ConversationMemory()
        
        # Load data
        self.df = pd.read_csv('star_wars_dataset.csv')
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
        """Train a new LSTM model if pre-trained one is not available"""
        patterns = self.df['pattern'].tolist()
        intents = self.df['intent'].tolist()
        
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
        
        # Build and train LSTM model (matches train.py)
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
        
        # Train model
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
        """Apply consistent preprocessing"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def get_response(self, text):
        # Use memory to contextualize the query
        contextualized_text = self.memory.contextualize_query(text)
        processed_text = self.preprocess_text(contextualized_text)
        
        if not processed_text:
            response = "Please ask me something about Star Wars!"
            self.memory.update_context(text, response)
            return response
        
        try:
            # Convert text to sequence
            seq = self.tokenizer.texts_to_sequences([processed_text])
            
            if not seq or len(seq[0]) == 0:
                response = self.rule_based_fallback(text)
                self.memory.update_context(text, response)
                return response
            
            # Pad sequence and get prediction
            padded = pad_sequences(seq, maxlen=self.max_length, padding='post')
            pred = self.model.predict(padded, verbose=0)
            
            intent_index = np.argmax(pred)
            confidence = np.max(pred)
            
            # Use different confidence thresholds based on intent
            confidence_threshold = 0.6  # Default threshold
            
            # Lower threshold for greetings and farewells
            if any(word in processed_text for word in ['hello', 'hi', 'hey', 'bye', 'goodbye']):
                confidence_threshold = 0.3
            
            # Use model response if confident enough
            if confidence < confidence_threshold:
                response = self.rule_based_fallback(text)
                self.memory.update_context(text, response)
                return response
            
            intent_tag = self.le.inverse_transform([intent_index])[0]
            
            # Special handling for follow-up questions
            if self.memory.current_topic and self.memory.current_entity:
                follow_up_indicators = ['what about', 'how about', 'tell me more', 'what else',
                                      'and', 'who is', 'what is', 'where is']
                if any(indicator in processed_text for indicator in follow_up_indicators):
                    # Try to redirect to the current topic
                    current_intent = f"ask_{self.memory.current_topic}"
                    if current_intent in self.le.classes_:
                        intent_tag = current_intent
            
            resp_list = self.df[self.df['intent']==intent_tag]['response'].tolist()
            
            # HYBRID RESPONSE GENERATION
            if random.random() < 0.5 and len(resp_list) > 1:
                generated = self.hybrid_generation(resp_list, intent_tag)
                if generated:
                    self.memory.update_context(text, generated)
                    return generated
            
            response = np.random.choice(resp_list) if resp_list else "I don't have a response for that."
            self.memory.update_context(text, response)
            return response
            
        except Exception as e:
            print(f"Error in model prediction: {e}")
            response = self.rule_based_fallback(text)
            self.memory.update_context(text, response)
            return response
    
    def hybrid_generation(self, responses: List[str], intent_tag: str) -> str:
        """Hybrid approach to response generation"""
        methods = [
            self.simple_combination,
            self.synonym_replacement, 
            self.pattern_mixing,
            self.template_based
        ]
        
        # Try each generation method in random order
        random.shuffle(methods)
        for method in methods:
            result = method(responses, intent_tag)
            if result and result not in responses:  # Ensure it's actually new
                return result
        
        return None
    
    def simple_combination(self, responses: List[str], intent_tag: str) -> str:
        """Combine parts of different responses"""
        if len(responses) < 2:
            return None
        
        resp1, resp2 = random.sample(responses, 2)
        words1, words2 = resp1.split(), resp2.split()
        
        if len(words1) > 3 and len(words2) > 3:
            # Combine first half of first response with second half of second
            split_idx = min(len(words1) // 2, len(words2) // 2)
            new_response = ' '.join(words1[:split_idx] + words2[split_idx:])
            return new_response.capitalize()
        
        return None
    
    def synonym_replacement(self, responses: List[str], intent_tag: str) -> str:
        """Replace words with synonyms"""
        base_response = random.choice(responses)
        
        # Star Wars specific synonyms
        synonyms = {
            'jedi': ['Jedi Knight', 'Force user', 'lightsaber wielder', 'Jedi Master'],
            'sith': ['Sith Lord', 'dark side user', 'Sith warrior', 'dark side practitioner'],
            'spaceship': ['starship', 'vessel', 'craft', 'ship', 'spacecraft'],
            'powerful': ['strong', 'mighty', 'formidable', 'potent', 'influential'],
            'uses': ['wields', 'employs', 'utilizes', 'harnesses', 'commands'],
            'force': ['the Force', 'this energy field', 'this power', 'the cosmic Force'],
            'lightsaber': ['energy sword', 'laser sword', 'Jedi weapon', 'elegant weapon'],
            'rebellion': ['Rebel Alliance', 'resistance', 'freedom fighters', 'alliance'],
            'empire': ['Galactic Empire', 'Imperial forces', 'the Empire', 'Imperial regime'],
            'droid': ['robot', 'automaton', 'mechanical being', 'artificial intelligence']
        }
        
        words = base_response.split()
        for i, word in enumerate(words):
            lower_word = word.lower()
            if lower_word in synonyms and random.random() < 0.4:
                words[i] = random.choice(synonyms[lower_word])
        
        return ' '.join(words)
    
    def pattern_mixing(self, responses: List[str], intent_tag: str) -> str:
        """Mix sentence patterns from different responses"""
        patterns = []
        for response in responses:
            # Simple pattern extraction based on structure
            if ' is ' in response:
                patterns.append(('is', response))
            elif ' has ' in response:
                patterns.append(('has', response))
            elif ' can ' in response:
                patterns.append(('can', response))
            elif ' was ' in response:
                patterns.append(('was', response))
        
        if len(patterns) >= 2:
            pattern1 = random.choice(patterns)
            pattern2 = random.choice([p for p in patterns if p != pattern1])
            
            # Pattern mixing logic
            if pattern1[0] == 'is' and pattern2[0] == 'has':
                subject = pattern1[1].split(' is ')[0]
                description = pattern1[1].split(' is ')[1].split('.')[0]
                capability = pattern2[1].split(' has ')[1].split('.')[0]
                return f"{subject} is {description} and has {capability}."
            
            elif pattern1[0] == 'is' and pattern2[0] == 'can':
                subject = pattern1[1].split(' is ')[0]
                description = pattern1[1].split(' is ')[1].split('.')[0]
                ability = pattern2[1].split(' can ')[1].split('.')[0]
                return f"{subject} is {description} and can {ability}."
        
        return None
    
    def template_based(self, responses: List[str], intent_tag: str) -> str:
        """Use template-based generation for specific intents"""
        templates = {
            'character_info': [
                "{name} is {description} who {action}.",
                "The character {name} is known for {description} and {action}.",
                "{name}: {description}. They are famous for {action}."
            ],
            'ship_info': [
                "The {ship} is {description} used for {purpose}.",
                "{ship} - {description} primarily used for {purpose}.",
                "Known as {ship}, this vessel is {description} and serves as {purpose}."
            ],
            'force_info': [
                "The Force is {description} that allows {capability}.",
                "{description}, the Force enables {capability}.",
                "The Force: {description}. It grants the ability to {capability}."
            ]
        }
        
        if intent_tag in templates:
            template = random.choice(templates[intent_tag])
            
            # Try to extract entities from responses (simplified)
            for response in responses:
                if ' is a ' in response:
                    parts = response.split(' is a ')
                    if len(parts) > 1:
                        name = parts[0]
                        rest = parts[1]
                        if ' who ' in rest:
                            description, action = rest.split(' who ', 1)
                            action = action.rstrip('.')
                            return template.format(name=name, description=description, action=action)
            
            # Fallback: use random words from responses
            all_words = ' '.join(responses).split()
            if len(all_words) >= 3:
                random_words = random.sample(all_words, min(5, len(all_words)))
                return template.format(
                    name=random_words[0] if len(random_words) > 0 else "Unknown",
                    description=random_words[1] if len(random_words) > 1 else "mysterious",
                    action=random_words[2] if len(random_words) > 2 else "act"
                )
        
        return None
    
    def rule_based_fallback(self, text):
        """Rule-based fallback if model fails"""
        text_lower = text.lower().strip()
        
        # Enhanced keyword matching
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
        
        # Check for specific phrases first
        for keyword, response in keyword_responses.items():
            if keyword in text_lower:
                return response
        
        # Contextual follow-up handling
        if self.memory.current_entity:
            follow_up_phrases = ['what about', 'how about', 'tell me more', 'what else',
                               'and', 'who is', 'what is', 'where is', 'when did']
            if any(phrase in text_lower for phrase in follow_up_phrases):
                return f"I'm not sure about that aspect of {self.memory.current_entity}. Try asking something more specific!"
        
        # General fallback
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
        self.setWindowTitle('Star Wars Chatbot (LSTM + Hybrid Generation + Memory)')
        self.setGeometry(100, 100, 800, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Title
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
        
        # Chat area
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
        
        # Input area
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
        
        # Clear memory button
        self.clear_button = QPushButton('🧠 Clear Memory')
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #4A235A;
                color: #FFFFFF;
                border: 2px solid #6C3483;
                border-radius: 8px;
                padding: 12px;
                font-family: Arial;
                font-size: 13px;
                font-weight: bold;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #6C3483;
                border-color: #6C3483;
            }
            QPushButton:pressed {
                background-color: #4A235A;
                border-color: #4A235A;
            }
        """)
        self.clear_button.clicked.connect(self.clear_memory)
        
        input_layout.addWidget(self.input_field, 4)
        input_layout.addWidget(self.send_button, 1)
        input_layout.addWidget(self.clear_button, 1)
        
        layout.addWidget(input_frame)
        
        # Status label
        self.status_label = QLabel("🤖 LSTM Model Loaded + Hybrid Generation + Memory - Ready to chat!")
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
        
        # Add welcome message
        self.add_message("Chatbot", "Hello! I'm a Star Wars chatbot powered by LSTM with hybrid response generation! 🌌")
        self.add_message("Chatbot", "I now have memory! Ask me about a character, then ask follow-up questions!")
        self.add_message("Chatbot", "Try: 'Tell me about Luke Skywalker' then 'What about his father?'")
    
    def clear_memory(self):
        self.chatbot.memory.reset()
        self.status_label.setText("🧠 Memory cleared! Starting fresh conversation...")
    
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
        
        # Check if the message contains specific phrases and play audio
        if sender == "Chatbot":
            if "General Kenobi!?" in message:
                self.play_general_kenobi_audio()
            elif "Do. Or do not. There is no try." in message:
                self.play_yoda_dodn_audio()    
            elif "Hello, there!" in message:
                self.play_hello_there_audio()    
            elif "I find your lack of faith disturbing." in message:
                self.play_lack_of_faith_audio()    
            elif "NOOO!" in message:
                self.play_nooo()
            elif "DO IT!" in message:
                self.play_do_it_audio()

    def play_general_kenobi_audio(self):
        """Play the General Kenobi audio"""
        try:
            pygame.mixer.music.load("sfx/general-kenobi.mp3")
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Could not play audio: {e}")

    def play_yoda_dodn_audio(self):
        """Play the Do or Do not audio"""
        try:
            pygame.mixer.music.load("sfx/yoda-do-or-do-not.mp3")
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Could not play audio: {e}")

    def play_hello_there_audio(self):
        """Play the Hello, there audio"""
        try:
            pygame.mixer.music.load("sfx/hello-there.mp3")
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Could not play audio: {e}")

    def play_lack_of_faith_audio(self):
        try:
            pygame.mixer.music.load("sfx/lack-of-faith.mp3")
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Could not play audio: {e}")

    def play_nooo(self):
        try:
            pygame.mixer.music.load("sfx/nooo.mp3")
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Could not play audio: {e}")

    def play_do_it_audio(self):
        try:
            pygame.mixer.music.load("sfx/do-it.mp3")
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Could not play audio: {e}")

    def send_message(self):
        message = self.input_field.text().strip()
        if not message:
            return
        
        self.add_message("You", message)
        self.input_field.clear()
        
        # Disable input while processing
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.status_label.setText("⏳ Processing your question...")
        
        # Process message in separate thread
        self.worker = ChatWorker(self.chatbot, message)
        self.worker.response_ready.connect(self.handle_response)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()
    
    def handle_response(self, response):
        self.add_message("Chatbot", response)
        # Re-enable input
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.status_label.setText("🤖 LSTM + Hybrid Generation + Memory - Ready to chat!")
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
