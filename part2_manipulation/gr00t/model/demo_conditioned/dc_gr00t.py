"""
DC-GR00T: Demo-Conditioned GR00T Model

Modified GR00T that conditions action prediction on task embeddings
extracted from demonstration videos.

Key changes from base GR00T:
1. Additional cross-attention to task embeddings from demo video
2. Task embedding is processed offline (from demo video)
3. Robot executes using its own observation (closed-loop)
4. Supports cross-embodiment learning
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.model.modules.dit import AlternateVLDiT, DiT
from gr00t.model.modules.eagle_backbone import EagleBackbone
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from .demo_encoder import DemoEncoder, VideoAlignmentLoss


@dataclass
class DCGr00tConfig(PretrainedConfig):
    """Configuration for Demo-Conditioned GR00T."""

    model_type: str = "DCGr00t"
    model_dtype: str = "bfloat16"

    # Base GR00T backbone config
    model_name: str = "nvidia/Eagle-Block2A-2B-v2"
    backbone_model_type: str = "eagle"
    backbone_embedding_dim: int = 2048
    tune_llm: bool = False
    tune_visual: bool = False
    select_layer: int = 16
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = True
    tune_top_llm_layers: int = 0  # Keep backbone frozen for DC mode

    # Action head config
    max_state_dim: int = 29
    max_action_dim: int = 29
    action_horizon: int = 16
    hidden_size: int = 1024
    input_embedding_dim: int = 1536

    # DiT config
    add_pos_embed: bool = True
    use_vlln: bool = True
    max_seq_len: int = 1024
    use_alternate_vl_dit: bool = True
    attend_text_every_n_blocks: int = 2

    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 32,
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
        }
    )

    # Flow matching config
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # Training config
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True
    state_dropout_prob: float = 0.0
    state_additive_noise_scale: float = 0.0
    max_num_embodiments: int = 32

    # ========== DC-specific config ==========
    # Demo encoder config
    demo_encoder_d_model: int = 768
    num_task_tokens: int = 16
    demo_temporal_layers: int = 4
    demo_resampler_layers: int = 2
    demo_nhead: int = 8
    demo_dim_feedforward: int = 2048
    demo_dropout: float = 0.1
    max_demo_frames: int = 64

    # Task conditioning config
    task_cross_attention_layers: int = 2
    fuse_task_with_language: bool = True  # Fuse task embedding with language

    # Training mode
    use_video_alignment_loss: bool = True
    alignment_loss_weight: float = 0.1

    def __init__(self, **kwargs):
        # Initialize diffusion_model_cfg with default if not provided
        if 'diffusion_model_cfg' not in kwargs:
            kwargs['diffusion_model_cfg'] = {
                "positional_embeddings": None,
                "num_layers": 32,
                "num_attention_heads": 32,
                "attention_head_dim": 48,
                "norm_type": "ada_norm",
                "dropout": 0.2,
                "final_dropout": True,
                "output_dim": 1024,
                "interleave_self_attention": True,
            }
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class TaskCrossAttention(nn.Module):
    """
    Cross-attention layer for conditioning on task embedding.

    Allows the robot's current observation features to attend to
    the task embedding extracted from the demonstration video.
    """

    def __init__(
        self,
        d_model: int,
        task_d_model: int,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project task embedding to same dimension if needed
        self.task_proj = (
            nn.Linear(task_d_model, d_model)
            if task_d_model != d_model
            else nn.Identity()
        )

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        task_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model] observation features
            task_embedding: [B, num_task_tokens, task_d_model]

        Returns:
            [B, seq_len, d_model] task-conditioned features
        """
        # Project task embedding
        task_emb = self.task_proj(task_embedding)

        # Cross-attention to task
        residual = x
        x = self.norm1(x)
        x, _ = self.cross_attn(query=x, key=task_emb, value=task_emb)
        x = residual + x

        # Self-attention
        residual = x
        x = self.norm2(x)
        x, _ = self.self_attn(query=x, key=x, value=x)
        x = residual + x

        # FFN
        residual = x
        x = self.norm3(x)
        x = residual + self.ffn(x)

        return x


class DCGr00tActionHead(nn.Module):
    """
    Action head with task conditioning for Demo-Conditioned GR00T.

    Extends the base GR00T action head with cross-attention to task embeddings.
    """

    supports_gradient_checkpointing = True

    def __init__(self, config: DCGr00tConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        # Base DiT model
        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )

        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        # State encoder
        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )

        # Action encoder/decoder
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        # VL layer norm
        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim)
            if config.use_vlln
            else nn.Identity()
        )

        # Position embedding
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(
                config.max_seq_len, self.input_embedding_dim
            )
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout
        self.state_dropout_prob = config.state_dropout_prob
        self.mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, self.input_embedding_dim))
            if self.state_dropout_prob > 0
            else None
        )

        # Flow matching params
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets

        # ========== DC-specific components ==========
        # Task cross-attention layers
        self.task_cross_attention = nn.ModuleList([
            TaskCrossAttention(
                d_model=config.backbone_embedding_dim,
                task_d_model=config.demo_encoder_d_model,
                nhead=config.demo_nhead,
                dropout=config.demo_dropout,
            )
            for _ in range(config.task_cross_attention_layers)
        ])

        # Task embedding projection for fusion
        self.task_to_backbone = nn.Linear(
            config.demo_encoder_d_model, config.backbone_embedding_dim
        )

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output_with_task(
        self,
        backbone_output: BatchFeature,
        task_embedding: torch.Tensor,
    ) -> BatchFeature:
        """
        Process backbone output and fuse with task embedding.

        Args:
            backbone_output: Output from Eagle backbone
            task_embedding: [B, num_task_tokens, d_model] from demo encoder

        Returns:
            Modified backbone output with task-conditioned features
        """
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)

        # Apply task cross-attention
        for cross_attn in self.task_cross_attention:
            backbone_features = cross_attn(backbone_features, task_embedding)

        # Optionally concatenate task tokens to backbone features
        if self.config.fuse_task_with_language:
            task_proj = self.task_to_backbone(task_embedding)
            backbone_features = torch.cat([backbone_features, task_proj], dim=1)

            # Update attention mask
            if "backbone_attention_mask" in backbone_output:
                task_mask = torch.ones(
                    task_embedding.shape[0], task_embedding.shape[1],
                    device=task_embedding.device, dtype=backbone_output["backbone_attention_mask"].dtype
                )
                backbone_output["backbone_attention_mask"] = torch.cat(
                    [backbone_output["backbone_attention_mask"], task_mask], dim=1
                )

            # Update image_mask to match backbone_attention_mask dimensions
            if "image_mask" in backbone_output:
                # Task tokens are non-image tokens, so extend image_mask with False
                task_image_mask = torch.zeros(
                    task_embedding.shape[0], task_embedding.shape[1],
                    device=task_embedding.device, dtype=backbone_output["image_mask"].dtype
                )
                backbone_output["image_mask"] = torch.cat(
                    [backbone_output["image_mask"], task_image_mask], dim=1
                )

        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        task_embedding: torch.Tensor,
    ) -> BatchFeature:
        """
        Forward pass with task conditioning.

        Args:
            backbone_output: From Eagle backbone (robot's current observation)
            action_input: State, action, embodiment_id, etc.
            task_embedding: [B, num_task_tokens, d_model] from demo encoder

        Returns:
            Loss and other outputs
        """
        # Process backbone with task embedding
        backbone_output = self.process_backbone_output_with_task(
            backbone_output, task_embedding
        )

        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device
        embodiment_id = action_input.embodiment_id

        # Encode state
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # State dropout
        if self.state_dropout_prob > 0 and self.training:
            do_dropout = (
                torch.rand(state_features.shape[0], device=device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

        # Flow matching
        actions = action_input.action
        noise = torch.randn(actions.shape, device=device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=device, dtype=actions.dtype)
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Position embedding
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Combine state and action
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask

        # Run DiT
        if self.config.use_alternate_vl_dit:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=backbone_output.get("image_mask"),
                backbone_attention_mask=backbone_output.get("backbone_attention_mask"),
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        # Decode actions
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1]:]

        # Loss
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        task_embedding: torch.Tensor,
    ) -> BatchFeature:
        """
        Generate actions conditioned on task embedding.
        """
        # Process backbone with task
        backbone_output = self.process_backbone_output_with_task(
            backbone_output, task_embedding
        )

        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device
        batch_size = vl_embeds.shape[0]
        embodiment_id = action_input.embodiment_id

        # Encode state
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Initialize actions as noise
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps

        # Denoising loop
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)

            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            sa_embs = torch.cat((state_features, action_features), dim=1)

            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.get("image_mask"),
                    backbone_attention_mask=backbone_output.get("backbone_attention_mask"),
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )

            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon:]

            actions = actions + dt * pred_velocity

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class DCGr00t(PreTrainedModel):
    """
    Demo-Conditioned GR00T: Learn from demonstration videos.

    The robot watches a demo (human/robot/Cosmos) to understand the task,
    then executes using its own observation (closed-loop control).
    """

    config_class = DCGr00tConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: DCGr00tConfig,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        super().__init__(config)
        self.config = config

        # Eagle backbone (processes robot's observation)
        self.backbone = EagleBackbone(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=True,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Demo encoder (processes demonstration video)
        self.demo_encoder = DemoEncoder(
            d_model=config.demo_encoder_d_model,
            num_task_tokens=config.num_task_tokens,
            temporal_layers=config.demo_temporal_layers,
            resampler_layers=config.demo_resampler_layers,
            nhead=config.demo_nhead,
            dim_feedforward=config.demo_dim_feedforward,
            dropout=config.demo_dropout,
            max_frames=config.max_demo_frames,
        )

        # Action head with task conditioning
        self.action_head = DCGr00tActionHead(config)

        # Video alignment loss for training
        if config.use_video_alignment_loss:
            self.alignment_loss = VideoAlignmentLoss()
        else:
            self.alignment_loss = None

        # Collator from base GR00T
        from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6DataCollator
        self.collator = Gr00tN1d6DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def encode_demo(
        self,
        demo_frames: torch.Tensor,
        demo_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode demonstration video to task embedding.

        Args:
            demo_frames: [B, T, H, W, C] demo video frames
            demo_type: [B] demo type (0=human, 1=robot, 2=cosmos, 3=own)

        Returns:
            [B, num_task_tokens, d_model] task embedding
        """
        return self.demo_encoder(demo_frames, demo_type)

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""
        if "vlm_content" in inputs:
            vlm_content_list = inputs["vlm_content"]
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = BatchFeature(data=inputs)

        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(
        self,
        inputs: dict,
        demo_frames: Optional[torch.Tensor] = None,
        demo_type: Optional[torch.Tensor] = None,
        task_embedding: Optional[torch.Tensor] = None,
    ) -> BatchFeature:
        """
        Training forward pass.

        Args:
            inputs: Robot observation, state, action, etc.
            demo_frames: [B, T, H, W, C] demo video (if not precomputed)
            demo_type: [B] demo type
            task_embedding: [B, N, D] precomputed task embedding (optional)

        Returns:
            Loss and outputs
        """
        # Get task embedding
        if task_embedding is None:
            if demo_frames is None:
                demo_frames = inputs.get("demo_frames")
            if demo_frames is None:
                raise ValueError("Either demo_frames or task_embedding must be provided")
            task_embedding = self.encode_demo(demo_frames, demo_type)

        # Process robot observation
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)

        # Get action loss
        outputs = self.action_head(backbone_outputs, action_inputs, task_embedding)

        # Add alignment loss if enabled
        if self.alignment_loss is not None and self.training:
            # Encode robot's own video as reference
            if "robot_demo_frames" in inputs:
                robot_embedding = self.encode_demo(inputs["robot_demo_frames"])
                align_loss = self.alignment_loss(task_embedding, robot_embedding)
                outputs["alignment_loss"] = align_loss
                outputs["loss"] = outputs["loss"] + self.config.alignment_loss_weight * align_loss

        return outputs

    @torch.no_grad()
    def get_action(
        self,
        inputs: dict,
        task_embedding: torch.Tensor,
    ) -> BatchFeature:
        """
        Inference: Get action conditioned on task embedding.

        Args:
            inputs: Current robot observation and state
            task_embedding: [B, N, D] task embedding from demo

        Returns:
            Predicted actions
        """
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        return self.action_head.get_action(backbone_outputs, action_inputs, task_embedding)

    @classmethod
    def from_pretrained_groot(
        cls,
        groot_path: str,
        config: Optional[DCGr00tConfig] = None,
        **kwargs,
    ):
        """
        Load DC-GR00T from pretrained GR00T checkpoint.

        Loads backbone and action head weights from GR00T,
        initializes demo encoder from scratch.
        """
        from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6

        # Force eager attention to avoid flash-attention requirement
        kwargs['attn_implementation'] = 'eager'

        # Load pretrained GR00T
        groot = Gr00tN1d6.from_pretrained(groot_path, **kwargs)

        # Create DC config
        if config is None:
            config = DCGr00tConfig()

        # Remove attn_implementation from kwargs before passing to DC model
        dc_kwargs = {k: v for k, v in kwargs.items() if k != 'attn_implementation'}

        # Create DC model
        dc_model = cls(config, **dc_kwargs)

        # Copy backbone weights
        dc_model.backbone.load_state_dict(groot.backbone.state_dict())
        print("Loaded backbone weights from GR00T")

        # Copy compatible action head weights
        groot_state = groot.action_head.state_dict()
        dc_state = dc_model.action_head.state_dict()

        copied = []
        for name, param in groot_state.items():
            if name in dc_state and dc_state[name].shape == param.shape:
                dc_state[name] = param
                copied.append(name)

        dc_model.action_head.load_state_dict(dc_state)
        print(f"Loaded {len(copied)} action head weights from GR00T")

        return dc_model

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register with HuggingFace
AutoConfig.register("DCGr00t", DCGr00tConfig)
AutoModel.register(DCGr00tConfig, DCGr00t)
