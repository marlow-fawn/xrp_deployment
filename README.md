
## Conversion
### For more details, please see the report

### `model_convert.py`

This script converts a pre-trained Soft Actor-Critic (SAC) model to a TensorFlow Lite model compatible with edge
devices. It:

1. Loads the trained SAC model.
2. Wraps the model with a DeterministicWrapper for deterministic behavior.
3. Converts the wrapped model to TensorFlow Lite format using ai_edge_torch and saves it as converted_model.tflite.

### `quantize.py`

This script applies 8-bit integer quantization to a TensorFlow SavedModel for efficient inference on low-power devices:

1. Loads the TensorFlow SavedModel (from `model_convert.py`).
2. Defines a representative dataset for quantization.
3. Converts the model to TensorFlow Lite format using TFLiteConverter with int8 quantization.
4. Saves the quantized model as quantized_model.tflite.


### `test.py`
Sanity check for the non-quantized model. 
1. Loads the converted_model.tflite
2. Sets a sample input (float32)
3. Runs inference
4. Prints the floating-point output

### `test_quant.py`
Sanity check for the quantized model. 
1. Loads the quantized_model.tflite 
2. Sets a sample input (int8)
3. Runs inference
4. Dequantizes
5. Prints floating-point output.

Output should *very roughly* match `test.py`