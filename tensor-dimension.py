import torch
import torch.nn.functional as F

x = torch.randn(4, 64, 7, 5)#batch,channel,height,width
print(x.amax(dim=(2, 3)).shape, x.mean(dim=(2, 3)).shape)  # height and weight are reduced to a single dimension
print(torch.allclose(x.amax(dim=(2, 3)), F.adaptive_max_pool2d(x, 1).flatten(1)))   
print(torch.allclose(x.mean(dim=(2, 3)), F.adaptive_avg_pool2d(x, 1).flatten(1)))   