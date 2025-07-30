import numpy as np
import tensorflow as tf

# Load the TFL model
interpreter = tf.lite.Interpreter(model_path="quantized_model.tflite")
interpreter.allocate_tensors()

# Prepare a sample input that matches the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
sample_input = np.zeros((1, 2), dtype=np.int8)
interpreter.set_tensor(input_details[0]['index'], sample_input)

# Run inference
interpreter.invoke()

# Check the output for sanity
output_data = interpreter.get_tensor(output_details[0]['index'])

# Dequantize the output based on the scale and zero point (which you can get from `output_details`)
scale, zero_point = output_details[0]['quantization']
dequantized_output = scale * (output_data - zero_point)
print("Output:", dequantized_output)