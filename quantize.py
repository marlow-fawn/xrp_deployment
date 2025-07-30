import tensorflow as tf
import numpy as np

def rep_dataset():
    for _ in range(100):
        data = np.random.rand(1, 2).astype(np.float32)
        yield [data]

# Load the model
converter = tf.lite.TFLiteConverter.from_saved_model("tf_model")

# Use the default optimization settings. These could be tweaked if the pi pico requires specific changes to the model
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Use the function to generate data as needed during quantizing
converter.representative_dataset = rep_dataset

# We needed INT8 for our target, as the pi pico isn't powerful enough for flota32.
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert and write the model
tflite_quant_model = converter.convert()
with open("quantized_model.tflite", "wb") as f:
    f.write(tflite_quant_model)