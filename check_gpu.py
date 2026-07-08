"""
GPU 诊断工具 — 检查 ROCm/CUDA 设备和 PyTorch 加速是否正常

用法:
  python check_gpu.py

输出:
  - GPU 型号 / 显存 / 驱动版本
  - PyTorch 与 GPU 连通性测试
  - 简单矩阵运算基准 (CPU vs GPU)
"""

import time
import torch
import numpy as np

print('=' * 55)
print('GPU Diagnosis for AMD Cloud (ROCm / CUDA)')
print('=' * 55)

# 1. PyTorch 版本
print(f'\n[1] PyTorch version: {torch.__version__}')

# 2. CUDA availability
cuda_avail = torch.cuda.is_available()
print(f'[2] torch.cuda.is_available(): {cuda_avail}')

if not cuda_avail:
    print('\n  ❌ GPU NOT DETECTED by PyTorch.')
    print('  Possible fixes:')
    print('  1) Check if ROCm PyTorch is installed:')
    print('     pip list | grep torch')
    print('  2) Reinstall ROCm PyTorch:')
    print('     pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/rocm5.6')
    print('  3) Check ROCm driver:')
    print('     rocminfo | grep "Name:"')
    print('     rocm-smi')
    exit(1)

# 3. GPU 属性
print(f'\n[3] GPU Properties:')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name}')
    print(f'    Compute Capability: {props.major}.{props.minor}')
    print(f'    Total Memory: {props.total_mem / 1024**3:.1f} GiB')
    print(f'    Multi Processors: {props.multi_processor_count}')

# 4. 当前设备
current = torch.cuda.current_device()
print(f'\n[4] Current device: {current} ({torch.cuda.get_device_name(current)})')

# 5. 显存信息
print(f'\n[5] Memory:')
print(f'  Reserved:  {torch.cuda.memory_reserved(current) / 1024**3:.2f} GiB')
print(f'  Allocated: {torch.cuda.memory_allocated(current) / 1024**3:.2f} GiB')
print(f'  Free:      {(torch.cuda.get_device_properties(current).total_mem - torch.cuda.memory_reserved(current)) / 1024**3:.2f} GiB')

# 6. ROCm version
print(f'\n[6] ROCm / CUDA version:')
print(f'  torch.version.cuda:  {torch.version.cuda}')
print(f'  torch.version.hip:   {torch.version.hip if hasattr(torch.version, "hip") else "N/A"}')

# 7. 张量运算测试
print(f'\n[7] Tensor operation test:')
try:
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.mm(x, y)
    result = z.sum().item()
    print(f'  ✅ Random matmul (1000x1000) on GPU: OK (sum={result:.2f})')
    del x, y, z
except Exception as e:
    print(f'  ❌ GPU tensor operation FAILED: {e}')

# 8. 模型训练测试
print(f'\n[8] Model train step test:')
try:
    model = torch.nn.Sequential(
        torch.nn.Linear(800, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4),
    ).to('cuda')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x = torch.randn(64, 800, device='cuda')
    y = torch.randint(0, 4, (64,), device='cuda')

    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f'  ✅ Model forward+backward on GPU: OK (loss={loss.item():.4f})')
    del model, x, y, output, loss
except Exception as e:
    print(f'  ❌ Model train step FAILED: {e}')

# 9. 性能对比 (CPU vs GPU)
print(f'\n[9] Speed benchmark (matrix 2000x2000):')
size = 2000

# CPU
t0 = time.time()
a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)
for _ in range(5):
    c_cpu = torch.mm(a_cpu, b_cpu)
cpu_time = (time.time() - t0) / 5

# GPU
t0 = time.time()
a_gpu = torch.randn(size, size, device='cuda')
b_gpu = torch.randn(size, size, device='cuda')
for _ in range(5):
    c_gpu = torch.mm(a_gpu, b_gpu)
torch.cuda.synchronize()
gpu_time = (time.time() - t0) / 5

print(f'  CPU: {cpu_time*1000:.1f} ms per matmul')
print(f'  GPU: {gpu_time*1000:.1f} ms per matmul')
ratio = cpu_time / gpu_time if gpu_time > 0 else float('inf')
print(f'  Speedup: {ratio:.1f}x')
if ratio > 2:
    print(f'  ✅ GPU is working and accelerating!')
else:
    print(f'  ⚠️  GPU speedup is low — check if model is on GPU')

print('\n' + '=' * 55)
print('GPU Diagnosis Complete.')
print('=' * 55)
