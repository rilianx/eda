import numpy as np
from copy import deepcopy
import random
import math

from .TSP_state import TSP_State
from .solveTSP_v2 import generate_random_points_and_distance_matrix, solve, plot_tour, calculate_tour_cost


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Softmax, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras import backend as K
from keras.utils  import to_categorical
import keras

class CustomMaskLayer(Layer):
    def call(self, inputs):
        # Generar máscara: 1 si el vector no es completamente ceros, 0 si lo es
        mask = tf.reduce_any(inputs != 0, axis=-1)
        mask = tf.expand_dims(tf.expand_dims(mask, 1), 1)
        return mask
    
def create_att_model(vec_len=6, num_heads=5, key_dim=32):

    input_layer = Input(shape=(None,vec_len))
    mask_layer = CustomMaskLayer()(input_layer)

    input_projection = input_layer #Dense(10, activation='tanh')(input_layer)

    att = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=0.2, output_shape=1)(
        query=input_projection, value=input_projection, key=input_projection,
            attention_mask=mask_layer)

    output = Softmax()(Lambda(lambda x: K.squeeze(x, -1))(att))
    model = Model(inputs=input_layer, outputs=output)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def edist(punto1, punto2):
    return math.sqrt((punto1[0] - punto2[0])**2 + (punto1[1] - punto2[1])**2)

# función para transformar un estado tsp en una secuencia de vectores
# para el modelo basado en capas de atención
def state2vecSeq(tsp_s):
    # creamos dos diccionarios para mantenre un mapeo de los
    # movimientos con los índices de la secuencia del modelo de aprendizaje

    idx2move = dict()
    move2idx = dict()
    origin = tsp_s.city_points[tsp_s.visited[0]]
    destination = tsp_s.city_points[tsp_s.visited[-1]]

    origin_dist = 0.0
    dest_dist = edist(origin, destination)

    seq = [list(origin) + [1,0] + [origin_dist, dest_dist]] # Última ciudad visitada (origen)


    idx2move[0] = ("constructive-move", tsp_s.visited[-1])
    move2idx[tsp_s.visited[-1]] = 0

    idx = 1
    for i in tsp_s.not_visited:
        point = list(tsp_s.city_points[i])
        origin_dist = edist( point, origin)
        dest_dist = edist( point, destination)
        if i == tsp_s.visited[0]:
          city_vector = point + [0, 1] + [origin_dist, 0.0]  # Ciudad final
        else:
          city_vector = point + [0, 0] + [origin_dist, dest_dist] # Otras ciudades
        seq.append(city_vector)
        idx2move[idx] = ("constructive-move", i)
        move2idx[i] = idx
        idx += 1

    return seq, idx2move, move2idx



def generate_data(max_cities=20, nb_sample=100):
    X = []  # Lista para almacenar las secuencias de entrada
    Y = []  # Lista para almacenar las etiquetas objetivo (las siguientes ciudades a visitar)
    seq_len = max_cities + 1  # Longitud de la secuencia, ajustada para incluir una ciudad extra

    # Bucle para generar datos hasta alcanzar el número deseado de muestras
    while True:
        # 1. Generamos instancia aleatoria
        n_cities = max_cities
        dim = 2  # Dimensión para las coordenadas de la ciudad (2D: x, y)
        city_points = np.random.rand(n_cities, dim)  # Generar puntos aleatorios para las ciudades

        # 2. Resolvemos TSP usando algoritmo tradicional
        TSP_State.initClass(city_points)
        initial_state = TSP_State([0])  # Estado inicial del TSP, empezando por la ciudad 0
        final_state = solve(city_points)  # Resolver el TSP y obtener un estado final

        # 3. Iteramos sobre los movimientos de la solución final para generar varias muestras:
        # estado (X) -> movimiento (Y)
        current_state = initial_state
        samples_per_sol = 5  # Número máximo de muestras por solución
        for move in final_state.moves:
            seq, _, move2idx = state2vecSeq(current_state)  # Convertir el estado actual a secuencia vectorizada
            seq_padded = np.zeros((seq_len, 6))  # Crear secuencia acolchada con ceros
            seq_padded[:len(seq), :] = np.array(seq)  # Rellenar con la secuencia real

            X.append(seq_padded)  # Añadir la secuencia a X
            Y.append(to_categorical(move2idx[move[1]], num_classes=seq_len))  # Añadir el movimiento como categoría a Y

            current_state.transition(move, evaluate=False)  # Hacer la transición al siguiente estado

            # Condiciones de parada basadas en el número de ciudades visitadas/no visitadas o muestras generadas
            if len(current_state.not_visited) < 3 or len(current_state.visited) > samples_per_sol or len(X) >= nb_sample:
                break

        # Romper el bucle externo si se ha alcanzado el número deseado de muestras
        if len(X) >= nb_sample:
            break

    # Devolver los datos como arrays de NumPy
    return np.stack(X), np.stack(Y)
