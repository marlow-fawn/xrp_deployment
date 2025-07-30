import numpy as np
import tensorflow as tf

# Load the TFL model
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Prepare a sample input that matches the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
sample_input = np.zeros((1, 2), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], sample_input)

# Run inference
interpreter.invoke()

# Check the output for sanity
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Output:", output_data)