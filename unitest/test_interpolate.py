import torch
import torch.nn.functional as F

device=torch.device("cuda:1")
kernel_cal = torch.randn(2, 1, 3, 2, 2).to(device)
original_kernel = kernel_cal.clone().detach()
kernel_cal = kernel_cal.squeeze(1).view(2, 3, -1)
kernel_cal = kernel_cal.permute(0, 2, 1)
kernel_cal = F.interpolate(kernel_cal, size=6, mode='linear', align_corners=True) # 2, 80*80*2, frames+1
kernel_cal = kernel_cal.permute(0, 2, 1) # 2, frames, 80*80*2
kernel_cal = kernel_cal.view(2, 1, 6, 2, 2) # 2, 1, frames, 80, 80, 2
kernel_cal = kernel_cal.contiguous()
#print(kernel_cal[:, :, -1, :, :, :] == original_kernel[:, :, -1, :, :, :])
print(original_kernel)
print(kernel_cal)
kernel_cal = torch.view_as_complex(kernel_cal)