"""
Demo Encoder: Extract Task Embeddings from Demonstration Videos

This module extracts embodiment-agnostic task representations from demo videos.
The demo can be:
- Human hand performing the task
- Another robot performing the task
- Cosmos-generated video of the task
- Your own robot from a different viewpoint

The encoder produces a fixed-size task embedding that captures:
- What objects are involved
- What is the goal state
- Motion patterns and trajectory hints

Architecture (based on Vid2Robot):
1. Frame Encoder: Encode individual frames using pretrained ViT
2. Temporal Transformer: Learn temporal dynamics across frames
3. Perceiver Resampler: Compress to fixed number of task tokens
"""

import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model] with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Transformer for learning temporal dynamics from video frames.

    Takes frame embeddings and learns motion patterns across time.
    """

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_frames: int = 64,
    ):
        super().__init__()

        self.d_model = d_model
        self.pos_encoding = TemporalPositionalEncoding(d_model, max_frames, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm for output
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        frame_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            frame_embeddings: [batch_size, num_frames, d_model]
            attention_mask: [batch_size, num_frames] optional mask

        Returns:
            [batch_size, num_frames, d_model] temporally-aware embeddings
        """
        # Add positional encoding
        x = self.pos_encoding(frame_embeddings)

        # Convert attention mask to transformer format if provided
        if attention_mask is not None:
            # Transformer expects True for positions to mask (ignore)
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        return x


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler: Compress variable-length video to fixed task tokens.

    Uses learned query tokens to attend to frame embeddings and produce
    a fixed-size task representation.

    Based on Flamingo's Perceiver Resampler architecture.
    """

    def __init__(
        self,
        d_model: int = 768,
        num_queries: int = 16,  # Number of output task tokens
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_queries = num_queries

        # Learned query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)

        # Cross-attention layers
        self.layers = nn.ModuleList([
            PerceiverResamplerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        frame_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            frame_embeddings: [batch_size, num_frames, d_model]
            attention_mask: [batch_size, num_frames] optional mask

        Returns:
            [batch_size, num_queries, d_model] task embedding tokens
        """
        batch_size = frame_embeddings.shape[0]

        # Expand query tokens for batch
        queries = self.query_tokens.expand(batch_size, -1, -1)

        # Apply cross-attention layers
        for layer in self.layers:
            queries = layer(queries, frame_embeddings, attention_mask)

        return self.norm(queries)


class PerceiverResamplerLayer(nn.Module):
    """Single layer of Perceiver Resampler with cross-attention and FFN."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        # Cross-attention: queries attend to frame embeddings
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)

        # Self-attention among query tokens
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        queries: torch.Tensor,
        frame_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            queries: [batch_size, num_queries, d_model]
            frame_embeddings: [batch_size, num_frames, d_model]
            attention_mask: [batch_size, num_frames]
        """
        # Convert mask format
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        # Cross-attention (queries attend to frames)
        x = self.cross_attn_norm(queries)
        attn_out, _ = self.cross_attn(
            query=x,
            key=frame_embeddings,
            value=frame_embeddings,
            key_padding_mask=key_padding_mask,
        )
        queries = queries + attn_out

        # Self-attention among queries
        x = self.self_attn_norm(queries)
        attn_out, _ = self.self_attn(query=x, key=x, value=x)
        queries = queries + attn_out

        # FFN
        x = self.ffn_norm(queries)
        queries = queries + self.ffn(x)

        return queries


class DemoEncoder(nn.Module):
    """
    Complete Demo Encoder: Video -> Task Embedding

    Takes a demonstration video and produces an embodiment-agnostic
    task embedding that captures what needs to be done.

    Architecture:
    1. Frame Encoder (ViT) -> per-frame embeddings
    2. Temporal Transformer -> motion-aware embeddings
    3. Perceiver Resampler -> fixed-size task tokens
    """

    def __init__(
        self,
        frame_encoder: Optional[nn.Module] = None,
        d_model: int = 768,
        num_task_tokens: int = 16,
        temporal_layers: int = 4,
        resampler_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_frames: int = 64,
        use_pretrained_encoder: bool = True,
        pretrained_model: str = "google/siglip-base-patch16-224",
    ):
        """
        Args:
            frame_encoder: Optional pretrained frame encoder (ViT)
            d_model: Hidden dimension
            num_task_tokens: Number of output task tokens
            temporal_layers: Number of temporal transformer layers
            resampler_layers: Number of perceiver resampler layers
            nhead: Number of attention heads
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
            max_frames: Maximum number of frames
            use_pretrained_encoder: Whether to use pretrained ViT
            pretrained_model: HuggingFace model name for pretrained encoder
        """
        super().__init__()

        self.d_model = d_model
        self.num_task_tokens = num_task_tokens
        self.max_frames = max_frames

        # Frame encoder (ViT)
        if frame_encoder is not None:
            self.frame_encoder = frame_encoder
            self.has_custom_encoder = True
        elif use_pretrained_encoder:
            self.frame_encoder = self._load_pretrained_encoder(pretrained_model)
            self.has_custom_encoder = False
        else:
            # Simple CNN encoder as fallback
            self.frame_encoder = self._build_simple_encoder(d_model)
            self.has_custom_encoder = False

        # Project frame encoder output to d_model.
        # Built eagerly here (not lazily in forward) so the parameters are
        # registered before the optimizer/checkpoint are created.
        encoder_output_dim = self._infer_encoder_output_dim(d_model)
        if encoder_output_dim != d_model:
            self.frame_proj = nn.Linear(encoder_output_dim, d_model)
        else:
            self.frame_proj = nn.Identity()

        # Temporal transformer
        self.temporal_transformer = TemporalTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=temporal_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_frames=max_frames,
        )

        # Perceiver resampler
        self.perceiver_resampler = PerceiverResampler(
            d_model=d_model,
            num_queries=num_task_tokens,
            nhead=nhead,
            num_layers=resampler_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Demo type embedding (human, robot, cosmos, etc.)
        self.demo_type_embedding = nn.Embedding(4, d_model)
        # 0: human, 1: other_robot, 2: cosmos, 3: own_robot

    def _infer_encoder_output_dim(self, default_dim: int) -> int:
        """Infer the feature dimension produced by the frame encoder.

        Falls back to ``default_dim`` (== d_model) for the simple CNN encoder,
        which already projects to d_model.
        """
        config = getattr(self.frame_encoder, "config", None)
        if config is not None:
            vision_config = getattr(config, "vision_config", None)
            if vision_config is not None and hasattr(vision_config, "hidden_size"):
                return vision_config.hidden_size
            if hasattr(config, "hidden_size"):
                return config.hidden_size
        return default_dim

    def _load_pretrained_encoder(self, model_name: str) -> nn.Module:
        """Load pretrained vision encoder from HuggingFace."""
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            # Freeze by default
            for param in model.parameters():
                param.requires_grad = False
            return model
        except Exception as e:
            print(f"Warning: Could not load {model_name}: {e}")
            print("Falling back to simple encoder")
            return self._build_simple_encoder(self.d_model)

    def _build_simple_encoder(self, d_model: int) -> nn.Module:
        """Build simple CNN encoder as fallback."""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, d_model),
        )

    def encode_frames(
        self,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode individual frames.

        Args:
            frames: [batch_size, num_frames, C, H, W] or [batch_size, num_frames, H, W, C]

        Returns:
            [batch_size, num_frames, d_model] frame embeddings
        """
        batch_size, num_frames = frames.shape[:2]

        # Ensure channel-first format
        if frames.shape[-1] == 3:
            frames = frames.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]

        # Flatten batch and time for encoding
        frames_flat = frames.reshape(batch_size * num_frames, *frames.shape[2:])

        # Normalize to [0, 1] if input is in [0, 255]
        if frames_flat.max() > 1.0:
            frames_flat = frames_flat / 255.0
        # SigLIP expects mean=0.5, std=0.5 normalization
        frames_flat = (frames_flat - 0.5) / 0.5

        # Encode
        if hasattr(self.frame_encoder, "vision_model"):
            # HuggingFace ViT
            outputs = self.frame_encoder.vision_model(frames_flat)
            embeddings = outputs.pooler_output  # [B*T, d]
        elif hasattr(self.frame_encoder, "get_image_features"):
            # CLIP-style
            embeddings = self.frame_encoder.get_image_features(frames_flat)
        else:
            # Custom encoder
            embeddings = self.frame_encoder(frames_flat)

        # Project to d_model
        embeddings = self.frame_proj(embeddings)

        # Reshape back
        embeddings = embeddings.reshape(batch_size, num_frames, -1)

        return embeddings

    def forward(
        self,
        demo_frames: torch.Tensor,
        demo_type: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract task embedding from demonstration video.

        Args:
            demo_frames: [batch_size, num_frames, H, W, C] or [B, T, C, H, W]
            demo_type: [batch_size] type of demo (0=human, 1=robot, 2=cosmos, 3=own)
            attention_mask: [batch_size, num_frames] mask for valid frames

        Returns:
            [batch_size, num_task_tokens, d_model] task embedding
        """
        batch_size = demo_frames.shape[0]

        # Encode individual frames
        frame_embeddings = self.encode_frames(demo_frames)  # [B, T, d_model]

        # Add demo type embedding if provided
        if demo_type is not None:
            type_emb = self.demo_type_embedding(demo_type)  # [B, d_model]
            frame_embeddings = frame_embeddings + type_emb.unsqueeze(1)

        # Apply temporal transformer
        temporal_embeddings = self.temporal_transformer(
            frame_embeddings, attention_mask
        )

        # Compress to task tokens
        task_embedding = self.perceiver_resampler(
            temporal_embeddings, attention_mask
        )

        return task_embedding

    def get_task_embedding(
        self,
        demo_video: torch.Tensor,
        demo_type: str = "robot",
        num_keyframes: int = 16,
    ) -> torch.Tensor:
        """
        Convenience method for inference.

        Args:
            demo_video: [T, H, W, C] single video (no batch)
            demo_type: "human", "robot", "cosmos", or "own"
            num_keyframes: Number of keyframes to sample

        Returns:
            [1, num_task_tokens, d_model] task embedding
        """
        # Sample keyframes
        T = demo_video.shape[0]
        indices = torch.linspace(0, T - 1, num_keyframes).long()
        keyframes = demo_video[indices]  # [num_keyframes, H, W, C]

        # Add batch dimension
        keyframes = keyframes.unsqueeze(0)  # [1, K, H, W, C]

        # Get demo type index
        type_map = {"human": 0, "robot": 1, "cosmos": 2, "own": 3}
        demo_type_idx = torch.tensor([type_map.get(demo_type, 1)], device=demo_video.device)

        # Encode
        with torch.no_grad():
            task_embedding = self.forward(keyframes, demo_type_idx)

        return task_embedding


class VideoAlignmentLoss(nn.Module):
    """
    Video Alignment Loss for cross-embodiment learning.

    Encourages same task -> similar embeddings, different task -> different embeddings.
    Uses temporal cycle consistency and contrastive learning.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        demo_embedding: torch.Tensor,
        robot_embedding: torch.Tensor,
        task_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute alignment loss between demo and robot embeddings.

        Args:
            demo_embedding: [B, N, D] demo task embeddings
            robot_embedding: [B, N, D] robot execution embeddings
            task_labels: [B] task IDs (for contrastive loss)

        Returns:
            Scalar loss value
        """
        # Pool to single vector per video
        demo_pooled = demo_embedding.mean(dim=1)  # [B, D]
        robot_pooled = robot_embedding.mean(dim=1)  # [B, D]

        # Normalize
        demo_pooled = F.normalize(demo_pooled, dim=-1)
        robot_pooled = F.normalize(robot_pooled, dim=-1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(demo_pooled, robot_pooled.T) / self.temperature

        # Diagonal should be positive pairs (same task)
        labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)

        # Cross-entropy loss (InfoNCE)
        loss_d2r = F.cross_entropy(sim_matrix, labels)
        loss_r2d = F.cross_entropy(sim_matrix.T, labels)

        loss = (loss_d2r + loss_r2d) / 2

        return loss
