import tensorflow as tf
import numpy as np
import string
from data import questions, answers

# Training Data
questions = [
    "¿Cómo estás?",
    "¿Cuál es tu nombre?",
    "¿Qué hora es?",
    "¿Cuál es el sentido de la vida?"
]
answers = [
    "Estoy bien, gracias.",
    "Mi nombre es Chatbot.",
    "Es hora de programar.",
    "El sentido de la vida es 42."
]

# Cleaning and Tokenize text function
def clean_and_tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

# Questions and answers tokenize
tokenized_questions = [clean_and_tokenize(question) for question in questions]
tokenized_answers = [clean_and_tokenize(answer) for answer in answers]

# Build vocab
vocab = set()
for question_tokens, answer_tokens in zip(tokenized_questions, tokenized_answers):
    vocab.update(question_tokens)
    vocab.update(answer_tokens)
vocab = sorted(vocab)
word_index = {word: index for index, word in enumerate(vocab)}
vocab_size = len(vocab)

# Get the data for the model 
max_length = max(max(len(question), len(answer)) for question, answer in zip(tokenized_questions, tokenized_answers))
def tokens_to_one_hot(tokens):
    one_hot = np.zeros((max_length, vocab_size), dtype=np.float32)
    for i, token in enumerate(tokens):
        if token in word_index:
            one_hot[i, word_index[token]] = 1.0
    return one_hot
X_train = np.array([tokens_to_one_hot(question_tokens) for question_tokens in tokenized_questions])
y_train = np.array([tokens_to_one_hot(answer_tokens) for answer_tokens in tokenized_answers])

# Definir y entrenar el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_length, vocab_size)),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.RepeatVector(max_length),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation='softmax'))
])
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X_train, y_train, epochs=100, batch_size=1)

# Generate answers function
def generate_answer(question):
    tokenized_question = clean_and_tokenize(question)
    one_hot_question = tokens_to_one_hot(tokenized_question)
    one_hot_question = np.expand_dims(one_hot_question, axis=0)
    one_hot_answer = model.predict(one_hot_question)[0]
    tokens_answer = [vocab[np.argmax(token)] for token in one_hot_answer]
    return ' '.join(tokens_answer)

# Chatbot assessment
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Adiós!")
        break
    chatbot_response = generate_answer(user_input)
    print("Chatbot:", chatbot_response)