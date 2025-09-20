# Star Wars Chatbot

An intelligent chatbot that answers questions about the Star Wars universe using a hybrid approach of LSTM neural networks and rule-based responses with conversation memory.

## Features

- **LSTM Model**: Trained on Star Wars dialogue data for intent classification
- **Hybrid Response Generation**: Combines model predictions with rule-based fallbacks
- **Conversation Memory**: Maintains context for follow-up questions
- **PyQt5 GUI**: Beautiful Star Wars-themed interface
- **Audio Responses**: Plays iconic Star Wars sound clips for certain responses
- **Data Augmentation**: Synonym replacement for better model training

## Project Structure

star-wars-chatbot/
├── train.py # Model training script
├── star_wars_dataset.csv # Training data
├── star_wars_chat.py # Main chatbot application
├── sfx/ # Sound effects directory
├── requirements.txt # Python dependencies
└── README.md


## Installation

1. Clone the repository:
```bash
git clone https://github.com/GabiSt/star-wars-chatbot.git
cd star-wars-chatbot


2. Install dependencies:

pip install -r requirements.txt

Add sound files to the sfx/ directory (see below for sources)


## Usage

1. Train the model (optional - pre-trained model included):

python train.py

2. Run the chatbot

python star_wars_chat.py


## Sound Effects

1. The chatbot uses audio clips for certain responses. You can add these files to the sfx/ directory:

    general-kenobi.mp3 - "General Kenobi" response

    yoda-do-or-do-not.mp3 - Yoda's "Do or do not" quote

    hello-there.mp3 - Obi-Wan's "Hello there"

    lack-of-faith.mp3 - Vader's "Lack of faith" line

    nooo.mp3 - Vader's "Nooo"

    do-it.mp3 - Palpatine's "Do it"


## Dataset

The star_wars_dataset.csv file contains questions and answers about:

    Characters (Luke Skywalker, Darth Vader, Yoda, etc.)

    Concepts (The Force, Jedi, Sith)

    Ships and locations

    Movie information


## Model Architecture

    Embedding layer (128 dimensions)

    Two LSTM layers (128 and 64 units)

    Dropout for regularization (0.5)

    Dense output layer with softmax activation

## Contributing

Feel free to contribute by:

    Adding more Star Wars questions and answers to the dataset

    Improving the response generation

    Enhancing the GUI

    Adding more sound effects

This project is for educational purposes. Star Wars is a trademark of Lucasfilm Ltd. and Disney.
