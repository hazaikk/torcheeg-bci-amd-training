"""
兼容性修复模块
=============
处理不同版本依赖库的兼容性问题。
在项目入口文件开头调用 apply_all_fixes() 即可。
"""

import warnings


def apply_all_fixes():
    """应用所有已知兼容性修复。"""
    _fix_scipy_signal()
    _suppress_warnings()


def _fix_scipy_signal():
    """修复新版 scipy 中移除的 scipy.signal.hann

    scipy >= 1.14 移除了 scipy.signal.hann、scipy.signal.hamming 等，
    但部分旧代码（如某些信号处理库）仍直接调用 scipy.signal.hann。
    这里用 scipy.signal.windows.hann 补上。
    """
    import scipy.signal

    # scipy.signal.hann → scipy.signal.windows.hann
    if not hasattr(scipy.signal, 'hann'):
        from scipy.signal.windows import hann
        scipy.signal.hann = hann
        print('[FIX] Applied scipy.signal.hann → scipy.signal.windows.hann')

    if not hasattr(scipy.signal, 'hamming'):
        from scipy.signal.windows import hamming
        scipy.signal.hamming = hamming

    if not hasattr(scipy.signal, 'blackman'):
        from scipy.signal.windows import blackman
        scipy.signal.blackman = blackman


def _suppress_warnings():
    """压制常见的非关键警告"""
    warnings.filterwarnings('ignore', message='.*TorchEEG.*')
    warnings.filterwarnings('ignore', message='.*scipy.*hann.*')
