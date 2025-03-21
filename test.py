import torch

print("CUDA 是否可用:", torch.cuda.is_available())  # 应该返回 True
print("GPU 数量:", torch.cuda.device_count())  # 应该返回 1 或更多
print("GPU 名称:", torch.cuda.get_device_name(0))  # 应该返回 RTX 3060 Ti
print("当前 GPU:", torch.cuda.current_device())  # 应该返回 0

