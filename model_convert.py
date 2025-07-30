import ai_edge_torch
import torch
from stable_baselines3 import SAC

class DeterministicWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        return self.policy._predict(obs, deterministic=True)

path = "sac_xrp_best.zip" # This is the full pytorch model

# This needs to match the hyperparameters used to train the model
custom_objects = {
    "lr_schedule": lambda _: 3e-4,
    "learning_rate": 3e-4,
}
# TFL requires a sample observation
sample_obs = torch.zeros(1, 2, dtype=torch.float32)  # or torch.rand(1, 2)

# Load the policy
model = SAC.load(path, custom_objects=custom_objects)
policy = model.policy
policy = policy.to("cpu")

# The original model was probabilistic. TFLM requires deterministic models.
wrapper = DeterministicWrapper(policy.eval())

# Compress the model
edge_model = ai_edge_torch.convert(
    wrapper.eval(),
    sample_args=(sample_obs,),
)

# Export the compressed model
edge_model.export("converted_model.tflite")

# Probably not needed, but left over from some experimenting
# torch.onnx.export(
#     wrapper,
#     sample_obs,
#     "model.onnx",
#     input_names=["input"],
#     output_names=["output"],
#     dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
#     opset_version=11,
# )
