'''
Bruce Chatbot
    Description:
    Easy Description: Bruce is a Chatbot that uses the USE as the Transformer (NN), a Clasification NN (model), and a dataset reveloped by Julian Principe 
    Author: Julian Principe, Universal-Sentence-Encoder
    Version: 0.1.0
'''


import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

#Training data
questions = ["¿Cómo estás?", "¿Cuál es tu nombre?", "¿Qué hora es?"]
answers = ["Estoy bien, gracias.", "Mi nombre es Bruce, soy un chatbot AI desarrollado por Julián Príncipe.", "Es hora de programar."]

#Load the pre-trained model Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

#Questions and Answers coding
questions_encoded = embed(questions)
answers_encoded = embed(answers)

#Simple Clasification Model
inputs = tf.keras.Input(shape=(512,))
outputs = tf.keras.layers.Dense(512, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(512, activation='relu')(outputs)
outputs = tf.keras.layers.Dense(512, activation='relu')(outputs)
outputs = tf.keras.layers.Dense(512, activation='relu')(outputs)
outputs = tf.keras.layers.Dense(512, activation='relu')(outputs)
outputs = tf.keras.layers.Dense(512, activation='relu')(outputs)
outputs = tf.keras.layers.Dense(len(answers), activation='softmax')(outputs)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

#Model training
model.fit(questions_encoded, np.arange(len(answers)), epochs=10, batch_size=1)

#Inference
while True:
    question = input("You: ")
    question_encoded = embed([question])
    answer_index = np.argmax(model.predict([question_encoded]))
    print("Chatbot:", answers[answer_index])