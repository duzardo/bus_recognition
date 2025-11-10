# -*- coding: utf-8 -*-
"""
Módulos do Sistema de Reconhecimento de Letreiros de Ônibus
"""

from .dataset_manager import DatasetManager
from .ocr_manager import OCRManager
from .text_similarity import TextSimilarity
from .utils import (
    safe_open_image,
    sanitize_filename,
    calculate_iou,
    setup_logging,
    MetricsCollector
)

__all__ = [
    'DatasetManager',
    'OCRManager',
    'TextSimilarity',
    'safe_open_image',
    'sanitize_filename',
    'calculate_iou',
    'setup_logging',
    'MetricsCollector'
]
