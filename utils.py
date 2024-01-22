import matplotlib.pyplot as plt

def plot_int_histogram(values):
  # Crear el histograma
  plt.hist(values, bins=range(min(values), max(values) + 2), edgecolor='black', align='left')

  # Añadir títulos y etiquetas
  plt.title('Histograma')
  plt.xlabel('Valores')
  plt.ylabel('Frecuencia')

  # Mostrar el histograma
  plt.show()

def plot_history(history):
  # Extracting the history records
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  # If your model includes precision, it will be stored like this
  # Make sure 'precision' matches the name of the metric used in your model
  precision = history.history['accuracy']
  val_precision = history.history['val_accuracy']

  epochs = range(1, len(loss) + 1)

  # Plotting training and validation loss
  plt.figure(figsize=(12, 5))

  plt.subplot(1, 2, 1)
  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  # Plotting training and validation precision
  plt.subplot(1, 2, 2)
  plt.plot(epochs, precision, 'b', label='Training Accuracy')
  plt.plot(epochs, val_precision, 'r', label='Validation Accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Precision')
  plt.legend()

  plt.tight_layout()
  plt.show()

def handle_multiple_states(func):
    def wrapper(*args, **kwargs):
        # Determinar cuál argumento es 'state'
        state_arg_name = 'state'  # asumiendo que todas las funciones nombran este argumento como 'state'
        state = args[0]

        # Función auxiliar para manejar un único estado
        def handle_single_state(single_state):
            # Reemplaza el argumento 'state' con el estado actual
            new_args = [single_state] + list(args[1:])
            return func(*new_args, **kwargs)

        # Verificar si 'state' es una lista o un único estado
        if isinstance(state, list):
            # Aplicar 'func' a cada estado en la lista
            return [handle_single_state(single_state) for single_state in state]
        else:

            # Aplicar 'func' a un único estado
            return handle_single_state(state)
    return wrapper


