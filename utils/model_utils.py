"""
模型工厂 — 初始化 TorchEEG CNN 模型
"""

import os
import sys
from typing import Union

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    config, EEGNetParams, FBCNetParams, FBMSNetParams,
    CSPNetParams, LMDAConfig, TSCeptionParams
)


def get_device() -> str:
    """检测可用设备: CUDA (NVIDIA) / ROCm (AMD) / CPU"""
    if not hasattr(get_device, '_cached'):
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            name = props.name.lower()
            if 'amd' in name or 'instinct' in name or 'radeon' in name:
                print(f'[DEVICE] AMD GPU detected: {props.name}')
            else:
                print(f'[DEVICE] NVIDIA GPU detected: {props.name}')
            get_device._cached = 'cuda'
        else:
            print('[DEVICE] No GPU found, using CPU')
            get_device._cached = 'cpu'
    return get_device._cached


def print_gpu_info(device: str):
    """打印详细的 GPU 诊断信息"""
    if device == 'cpu':
        print('  [GPU] No GPU available — running on CPU')
        return

    print(f'  [GPU] Device: {torch.cuda.get_device_name(0)}')
    print(f'  [GPU] Compute Capability: {torch.cuda.get_device_capability(0)}')
    try:
        total = torch.cuda.get_device_properties(0).total_mem / 1024 ** 3
        reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
        allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
        free = total - reserved
        print(f'  [GPU] Memory: {total:.1f}GiB total, '
              f'{free:.1f}GiB free, {allocated:.1f}GiB allocated')
    except Exception:
        pass
    print(f'  [GPU] PyTorch CUDA version: {torch.version.cuda}')
    print(f'  [GPU] ROCm: {torch.version.hip if hasattr(torch.version, "hip") else "N/A"}')

    # 验证 GPU 可用
    try:
        t = torch.tensor([1.0, 2.0, 3.0], device=device)
        t = t + 1
        print(f'  [GPU] Verify: tensor device check OK ({t.device})')
    except Exception as e:
        print(f'  [GPU] Verify FAILED: {e}')


def verify_device(device: str) -> str:
    """验证设备可用，打印诊断信息，返回可靠设备名"""
    print_gpu_info(device)

    if device == 'cpu':
        return 'cpu'

    # 确认模型能正常放到 GPU
    try:
        test_model = torch.nn.Linear(10, 10).to(device)
        test_input = torch.randn(4, 10).to(device)
        test_output = test_model(test_input)
        assert test_output.device.type == device, \
            f'Expected {device}, got {test_output.device}'
        del test_model, test_input, test_output
        return device
    except Exception as e:
        print(f'  [DEVICE] GPU failed, falling back to CPU: {e}')
        return 'cpu'


def get_criterion(model_name: str) -> nn.Module:
    """根据模型输出类型选择损失函数"""
    # FBCNet / FBMSNet 输出 LogSoftmax → NLLLoss
    if model_name in ['FBCNet', 'FBMSNet', 'CSPNet']:
        return nn.NLLLoss()
    else:
        return nn.CrossEntropyLoss()


def create_model(model_name: str,
                 num_classes: int = 4,
                 chunk_size: int = 800,
                 num_electrodes: int = 22,
                 in_channels: int = 1,
                 **kwargs) -> nn.Module:
    """创建指定模型实例"""
    try:
        import torcheeg.models as tm
    except ImportError:
        raise ImportError(
            'torcheeg not installed. Run: pip install torcheeg')

    if model_name == 'EEGNet':
        params = EEGNetParams()
        return tm.EEGNet(
            chunk_size=chunk_size,
            num_electrodes=num_electrodes,
            F1=kwargs.get('F1', params.F1),
            F2=kwargs.get('F2', params.F2),
            D=kwargs.get('D', params.D),
            kernel_1=kwargs.get('kernel_1', params.kernel_1),
            kernel_2=kwargs.get('kernel_2', params.kernel_2),
            dropout=kwargs.get('dropout', params.dropout),
            num_classes=num_classes
        )

    elif model_name == 'FBCNet':
        params = FBCNetParams()
        in_ch = kwargs.get('in_channels', 9)
        return tm.FBCNet(
            num_electrodes=num_electrodes,
            chunk_size=chunk_size,
            in_channels=in_ch,
            num_S=kwargs.get('num_S', params.num_S),
            num_classes=num_classes,
            temporal=kwargs.get('temporal', params.temporal),
            stride_factor=kwargs.get('stride_factor', params.stride_factor)
        )

    elif model_name == 'FBMSNet':
        params = FBMSNetParams()
        in_ch = kwargs.get('in_channels', 9)
        return tm.FBMSNet(
            num_electrodes=num_electrodes,
            chunk_size=chunk_size,
            in_channels=in_ch,
            num_classes=num_classes,
            stride_factor=kwargs.get('stride_factor', params.stride_factor),
            temporal=kwargs.get('temporal', params.temporal),
            num_feature=kwargs.get('num_feature', params.num_feature),
            dilatability=kwargs.get('dilatability', params.dilatability)
        )

    elif model_name == 'CSPNet':
        params = CSPNetParams()
        return tm.CSPNet(
            chunk_size=chunk_size,
            num_electrodes=num_electrodes,
            num_classes=num_classes,
            dropout=kwargs.get('dropout', 0.5),
            num_filters_t=kwargs.get('num_filters_t', params.num_filters_t),
            filter_size_t=kwargs.get('filter_size_t', params.filter_size_t),
            num_filters_s=kwargs.get('num_filters_s', params.num_filters_s),
            pool_size_1=kwargs.get('pool_size_1', params.pool_size_1),
            pool_stride_1=kwargs.get('pool_stride_1', params.pool_stride_1),
        )

    elif model_name == 'LMDA':
        params = LMDAConfig()
        return tm.LMDA(
            num_electrodes=num_electrodes,
            chunk_size=chunk_size,
            num_classes=num_classes,
            depth=kwargs.get('depth', params.depth),
            kernel=kwargs.get('kernel', params.kernel),
            hid_channels_1=kwargs.get('hid_channels_1', params.hid_channels_1),
            hid_channels_2=kwargs.get('hid_channels_2', params.hid_channels_2),
            pool_size=kwargs.get('pool_size', params.pool_size),
        )

    elif model_name == 'TSCeption':
        params = TSCeptionParams()
        return tm.TSCeption(
            num_electrodes=num_electrodes,
            in_channels=in_channels,
            num_classes=num_classes,
            sampling_rate=250,
            num_T=kwargs.get('num_T', params.num_T),
            num_S=kwargs.get('num_S', params.num_S),
            hid_channels=kwargs.get('hid_channels', params.hid_channels),
            dropout=kwargs.get('dropout', params.dropout),
        )

    else:
        raise ValueError(
            f'Unknown model: {model_name}. '
            f'Available: EEGNet, FBCNet, FBMSNet, '
            f'CSPNet, LMDA, TSCeption')
