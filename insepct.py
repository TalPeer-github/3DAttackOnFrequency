import torch
print(torch.cuda.current_device())        # should be 0 (your assigned GPU)
print(torch.cuda.get_device_name(0))      # name of the GPU you're using
print(torch.cuda.device_count())          # how many GPUs are visible (should be 1)
