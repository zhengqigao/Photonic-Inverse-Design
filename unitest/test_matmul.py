import torch
from pyutils.general import TimerCtx, print_stat, TorchTracemalloc
from pyutils.torch_train import set_torch_deterministic
from core.models.layers.pac import nd2col
from core.models.layers.triton_matmul import matmul
from core.models.layers.triton_matmul2 import matmul as matmul2
from core.models.layers.triton_matmul_pac import matmul_forward as matmul_pac


def test_matmul():
    device = "cuda:0"
    set_torch_deterministic(0)
    x = torch.randn(64, 641, device=device)
    f = torch.randn(641, 64, device=device)
    for _ in range(5):
        y = matmul(x, f)

    torch.cuda.synchronize()
    with TimerCtx() as t:
        for _ in range(5):
            y = matmul(x, f)
        torch.cuda.synchronize()
    print(t.interval / 5)


    for _ in range(5):
        y = matmul2(x, f)

    torch.cuda.synchronize()
    with TimerCtx() as t:
        for _ in range(5):
            y = matmul2(x, f)
        torch.cuda.synchronize()
    print(t.interval / 5)

    for _ in range(5):
        y = torch.matmul(x, f)
    torch.cuda.synchronize()
    with TimerCtx() as t:
        for _ in range(5):
            y = torch.matmul(x, f)
        torch.cuda.synchronize()
    print(t.interval / 5)

def test_matmul_pac():
    device = "cuda:0"
    x = torch.randn(2,32,64,64, device=device, requires_grad=True)
    x = nd2col(x, 17, 1, 8)
    print(x.shape)
    w = torch.randn(32,32,17,17, device=device, requires_grad=True)
    k = torch.randn(1,1,64,64, device=device, requires_grad=True)
    k = nd2col(k, 17, 1, 8)
    with TorchTracemalloc(True) as tracemalloc:
        y = matmul_pac(x, w, k)
    torch.cuda.empty_cache()
    y.sum().backward()

    with TorchTracemalloc(True) as tracemalloc:
        y = torch.einsum('ijklmn,ojkl->iomn', (x * k, w))
    torch.cuda.empty_cache()


    for _ in range(5):
        y = matmul_pac(x, w, k)
    
    torch.cuda.synchronize()
    with TimerCtx() as t:
        for _ in range(5):
            y = matmul_pac(x, w, k)
        torch.cuda.synchronize()
    print(t.interval / 5)


    for _ in range(5):
        y2 = torch.einsum('ijklmn,ojkl->iomn', (x * k, w))
    
    torch.cuda.synchronize()
    with TimerCtx() as t:
        for _ in range(5):
            y2 = torch.einsum('ijklmn,ojkl->iomn', (x * k, w))
        torch.cuda.synchronize()
    print(t.interval / 5)
    print(y[0,0,0])
    print(y2[0,0,0])
    assert torch.allclose(y, y2, atol=1e-6)
    
def test_backward():
    device = "cuda:0"
    x = torch.randn(2,32,64,64, device=device, requires_grad=True)
    x = nd2col(x, 17, 1, 8)
    print(x.shape)
    w = torch.randn(32,32,17,17, device=device, requires_grad=True)
    k = torch.randn(1,1,64,64, device=device, requires_grad=True)
    k = nd2col(k, 17, 1, 8)

    for _ in range(5):
        y = (x * k).sum(1)
    
    torch.cuda.synchronize()
    with TimerCtx() as t:
        for _ in range(5):
            y = (x * k).sum(1)
        torch.cuda.synchronize()
    print(t.interval / 5)

    for _ in range(5):
        y = torch.einsum('ijklmn,ijklmn->iklmn', (x, x))
    
    torch.cuda.synchronize()
    with TimerCtx() as t:
        for _ in range(5):
            y = torch.einsum('ijklmn,ijklmn->iklmn', (x, x))
        torch.cuda.synchronize()
    print(t.interval / 5)


if __name__ == "__main__":
    # test_matmul()
    # test_matmul_pac()
    test_backward()