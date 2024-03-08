import torch.nn as nn
from chop.models.utils import MaseModelInfo
from .vgg_cifar import get_vgg7


# PHYSICAL_MODELS = {
#     "jsc-toy": {
#         "model": get_vgg7,
#         "info": MaseModelInfo(
#             model_source="physical",
#             task_type="physical",
#             physical_data_point_classification=True,
#             is_fx_traceable=True,
#         ),
#     },
# }