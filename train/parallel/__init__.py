"""
并行训练模块

包含数据收集 Worker 和管理器
"""

from .worker import WorkerManager, worker_process

__all__ = [
    'WorkerManager',
    'worker_process'
]

