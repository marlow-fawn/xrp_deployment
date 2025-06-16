import google.protobuf
print(google.protobuf.__file__)

import ai_edge_torch
#import torch
from stable_baselines3 import SAC

path = "/home/marlow/Downloads/sac_xrp_best.zip"

custom_objects = {
    "lr_schedule": lambda _: 3e-4,
    "learning_rate": 3e-4,  # also good to include explicitly
}

model = SAC.load(path, custom_objects=custom_objects)
policy = model.policy
print(policy.eval())
edge_model = ai_edge_torch.convert(policy.eval())
