from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds

import math
import numpy as np
import matplotlib.pyplot as plt
import logging

from urllib import parse
from http.server import HTTPServer, BaseHTTPRequestHandler
logger = tf.get_logger()

logger.setLevel(logging.ERROR)


dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = [
    'Cero', 'Uno', 'Dos', 'Tres', 'Cuatro', 'Cinco', 'Seis',
    'Siete', 'Ocho', 'Nueve'
]

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

#Normalizar: Numeros de 0 a 255, que sean de 0 a 1
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

#Estructura de la red
model = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape=(28,28,1)),
	tf.keras.layers.Dense(72, activation=tf.nn.relu),
	tf.keras.layers.Dense(72, activation=tf.nn.relu),
	tf.keras.layers.Dense(10, activation=tf.nn.softmax) #para clasificacion
])

#Indicar las funciones a utilizar
model.compile(
	optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy']
)

#Aprendizaje por lotes de 32 cada lote
BATCHSIZE = 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCHSIZE)
test_dataset = test_dataset.batch(BATCHSIZE)

#Realizar el aprendizaje
model.fit(
	train_dataset, epochs=5,
	steps_per_epoch=math.ceil(num_train_examples/BATCHSIZE) #No sera necesario pronto
)

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

	def do_post(self):
		print("Peticion recibida")

		#Obtener datos de la peticion por ajax
		content_length = int(self.headers['content-Length'])	
		data = self.rfile.read(content_length)

		data = data.decode().replace('pixeles=', '')
		data = parse.unquote(data)

		arr = np.fromstring(data, np.float32, sep=",")
		arra = arr.reshape(28,28)
		arr = np.array(arr)
		arr = arr.reshape(1,28,28,1)

		prediction_values = model.predict(arr, batch_size=1)
		prediction = str(np.argmax(prediction_values))
		print("Prediccion final: " + prediction)

		self.send_response(200)
		#Evitar problemas con CORS
		self.send_header("Access-Control-Allow-Origin", "*")
		self.end_headers()
		self.wfile.write(prediction.encode())

print("Iniciando el servidor...")
server = HTTPServer(('localhost, 8000'), SimpleHTTPRequestHandler)
server.serve_forever()