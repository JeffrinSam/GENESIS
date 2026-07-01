"""
FlowDiT V3: Intelligent Video-Guided Navigation with Real-Time Adaptation

Clean implementation directory - separate from V2 to avoid confusion.
"""

__version__ = "3.0.0"
__author__ = "FlowDiT V3 Team"

from .models.flowdit_v3 import (
    FlowDiTv3,
    FlowDiTv3Config,
    create_flowdit_v3
)

__all__ = [
    'FlowDiTv3',
    'FlowDiTv3Config',
    'create_flowdit_v3',
]
