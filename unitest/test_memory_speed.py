import torch
import torch.nn as nn
from pyutils.general import TimerCtx, print_stat, TorchTracemalloc
from core.models.layers import PacConv2d
import numpy as np
import tqdm
def build_toy_cnn_model(kernel_size, device):
    padding = kernel_size // 2
    model = nn.Sequential(
        nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
    ).to(device)
    return model

class ToyPac(nn.Module):
    def __init__(self, kernel_size) -> None:
        super(ToyPac, self).__init__()
        padding = kernel_size // 2
        self.Pac1 = PacConv2d(32, 32, kernel_size=kernel_size, padding=padding, native_impl=True)
        self.Pac2 = PacConv2d(32, 32, kernel_size=kernel_size, padding=padding, native_impl=True)
        self.guidance = torch.randn(1, 8, 256, 256, device="cuda:0")
    
    def forward(self, x):
        x = self.Pac1(x, self.guidance).relu()
        x = self.Pac2(x, self.guidance).relu()
        return x

def test_memory_and_speed(model_type, device, step = 4, to_kernel: int = 75):
    inference_speed = []
    training_speed = []
    dummy_data = torch.randn(1, 32, 256, 256, device=device)
    kernel_sizes = range(3, to_kernel, step)
    for kernel_size in tqdm.tqdm(kernel_sizes):
        if model_type == "simple":
            model = build_toy_cnn_model(kernel_size, device)
        elif model_type == "pac":
            model = ToyPac(kernel_size).to(device)
        else:
            raise ValueError(f"model type {model_type} not recognized")
        # first test the speed in inference
        with torch.no_grad():
            for _ in range(6):
                y = model(dummy_data).detach()
        torch.cuda.synchronize()
        with torch.no_grad():
            with TimerCtx() as t:
                for _ in range(3):
                    y = model(dummy_data).detach()
                torch.cuda.synchronize()
        inference_time = t.interval / 3
        inference_speed.append(inference_time)
        # next test the speed in training
        torch.cuda.synchronize()
        with TimerCtx() as t:
            for _ in range(3):
                y = model(dummy_data).detach()
            torch.cuda.synchronize()
        training_time = t.interval / 3
        training_speed.append(training_time)
        del model
        torch.cuda.empty_cache()
    inference_memory = []
    training_memory = []
    for kernel_size in tqdm.tqdm(kernel_sizes):
        # next we test the memory usage in inference and train:
        # begin with inference
        if model_type == "simple":
            model = build_toy_cnn_model(kernel_size, device)
        elif model_type == "pac":
            model = ToyPac(kernel_size).to(device)
        with torch.no_grad():
            with TorchTracemalloc(False) as tracemalloc_inference:
                y = model(dummy_data).detach()
            inference_memory.append(tracemalloc_inference.peaked)
            del y
            torch.cuda.empty_cache()
        with TorchTracemalloc(False) as tracemalloc_training:
            y = model(dummy_data).detach()
        training_memory.append(tracemalloc_training.peaked)
        del y
        del model
        torch.cuda.empty_cache()
    
    inference_memory = np.array(inference_memory)
    training_memory = np.array(training_memory)
    inference_speed = np.array(inference_speed)
    training_speed = np.array(training_speed)

    table = np.stack([training_memory, inference_memory, training_speed, inference_speed], axis=1)

    np.savetxt(f"./unitest/toy_{model_type}_memory_speed.csv", table, delimiter=",", header="training_memory,inference_memory,training_speed,inference_speed", comments="")

    return None

if __name__ == "__main__":
    device = torch.device("cuda:0")
    test_memory_and_speed("pac", device, to_kernel=23, step=2)
    test_memory_and_speed("simple", device, to_kernel=23, step=2)
    
