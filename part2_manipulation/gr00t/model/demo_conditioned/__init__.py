"""
Demo-Conditioned GR00T (DC-GR00T)

A robot learning framework that enables learning from demonstration videos.
The robot watches a demo (human hand, another robot, Cosmos-generated, etc.)
to understand the task, then executes using its own sensors (closed-loop).

Key components:
- DemoEncoder: Extracts task embeddings from demonstration videos
- DCGr00t: Modified GR00T model with cross-attention to task embeddings
- DCPolicy: Inference pipeline for real-world deployment

Based on research from:
- Vid2Robot (Google): End-to-end video-conditioned policy learning
- CrossFormer: Cross-embodiment robot learning
- Track2Act: Point tracking for manipulation
"""

from .demo_encoder import DemoEncoder, TemporalTransformer, PerceiverResampler
from .dc_gr00t import DCGr00t, DCGr00tConfig
from .dc_policy import DCPolicy

__all__ = [
    "DemoEncoder",
    "TemporalTransformer",
    "PerceiverResampler",
    "DCGr00t",
    "DCGr00tConfig",
    "DCPolicy",
]
