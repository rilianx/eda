import matplotlib.pyplot as plt

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