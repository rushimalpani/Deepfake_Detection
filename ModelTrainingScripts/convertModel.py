import tensorflow as tf

# Load the TensorFlow protobuf model (.pb)
pb_model_path = "D:/WCE/TY/Sem-6/Project/trainedModel"  # Update this path to the correct one
loaded_pb_model = tf.saved_model.load(pb_model_path)

# Convert the TensorFlow protobuf model to a Keras model
loaded_keras_model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(None, None, 3)),  # Input shape may vary, adjust as needed
    loaded_pb_model.signatures["serving_default"]
])

# Save the Keras model as an HDF5 file
h5_model_path = "./Newmodel.h5"
loaded_keras_model.save(h5_model_path)
