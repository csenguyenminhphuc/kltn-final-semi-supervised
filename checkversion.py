# import torch
# print("torch:", torch.__version__)
# print("torch build cuda:", torch.version.cuda)
# print("cuda available:", torch.cuda.is_available())
# print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
# print("Num GPUs:", torch.cuda.device_count())

import cv2, ctypes.util as u
import numpy as np
import mmcv
import mmengine
print("mmengine:", mmengine.__version__)
print("mmcv:", mmcv.__version__)
print("numpy:", np.__version__)
print("cv2:", cv2.__version__)
print("libGL path:", u.find_library("GL"))
