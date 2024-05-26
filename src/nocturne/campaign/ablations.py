"""Five-family ablation matrix execution engine fulfilling Gate BA10 and capability B26."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nocturne.campaign.benchmark import BenchmarkSuiteRunner
from nocturne.models.attention.config import AttentionConfig
from nocturne.models.attention.model import SpatiotemporalAttentionModel
from nocturne.models.vldm.config import VLDMConfig
from nocturne.models.vldm.model import VariationalLatentDynamicsModel
from nocturne.models.likelihoods import PoissonLoss
from nocturne.evaluation.bits_per_spike import BitsPerSpikeEvaluator
from nocturne.evaluation.metrics import EvaluationMetricsCalculator


class AblationFamily(str, Enum):
    """The 5 standardized ablation study families."""

    OBJECTIVE = "family_1_objective"
    INPUT_MASKING = "family_2_input_masking"
    DEPTH_CAPACITY = "family_3_depth_capacity"
    TEMPORAL_CAUSALITY = "family_4_temporal_causality"
    DYNAMICS_DRIVE = "family_5_dynamics_drive"


@dataclass(frozen=True)
class AblationConfig:
    """Configuration for an individual ablation experiment."""

    family: AblationFamily
    variant_name: str
    dataset_name: str
    seed: int = 42
    num_steps: int = 20
    learning_rate: float = 1e-3
    device: str = "cpu"


@dataclass(frozen=True)
class AblationResult:
    """Quantitative evaluation outcome for an ablation experiment."""

    config: AblationConfig
    bps_held_in: Optional[float]
    bps_held_out: Optional[float]
    poisson_deviance: float
    pseudo_r2: float
    duration_s: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["config"]["family"] = self.config.family.value
        return d


class AblationRunner:
    """Executes the complete 5-family ablation matrix with multi-seed replication."""

    @classmethod
    def run_objective_ablation(
        cls,
        dataset_name: str = "mc_maze",
        seeds: Tuple[int, ...] = (42, 43, 44),
        num_steps: int = 20,
    ) -> List[AblationResult]:
        """Family 1: Poisson vs MSE vs Gaussian loss."""
        variants = ["poisson", "mse", "gaussian"]
        results = []

        for variant in variants:
            for s in seeds:
                cfg = AblationConfig(
                    family=AblationFamily.OBJECTIVE,
                    variant_name=variant,
                    dataset_name=dataset_name,
                    seed=s,
                    num_steps=num_steps,
                )
                res = cls._run_single_ablation(cfg)
                results.append(res)
        return results

    @classmethod
    def run_masking_ablation(
        cls,
        dataset_name: str = "mc_maze",
        seeds: Tuple[int, ...] = (42, 43, 44),
        num_steps: int = 20,
    ) -> List[AblationResult]:
        """Family 2: Co-smoothing vs unmasked vs token dropout."""
        variants = ["co_smoothing", "unmasked", "token_dropout_0.2"]
        results = []

        for variant in variants:
            for s in seeds:
                cfg = AblationConfig(
                    family=AblationFamily.INPUT_MASKING,
                    variant_name=variant,
                    dataset_name=dataset_name,
                    seed=s,
                    num_steps=num_steps,
                )
                res = cls._run_single_ablation(cfg)
                results.append(res)
        return results

    @classmethod
    def run_capacity_ablation(
        cls,
        dataset_name: str = "mc_maze",
        seeds: Tuple[int, ...] = (42, 43, 44),
        num_steps: int = 20,
    ) -> List[AblationResult]:
        """Family 3: Layer depth and embedding dimension scaling."""
        variants = ["depth1_dim32", "depth2_dim64", "depth4_dim64"]
        results = []

        for variant in variants:
            for s in seeds:
                cfg = AblationConfig(
                    family=AblationFamily.DEPTH_CAPACITY,
                    variant_name=variant,
                    dataset_name=dataset_name,
                    seed=s,
                    num_steps=num_steps,
                )
                res = cls._run_single_ablation(cfg)
                results.append(res)
        return results

    @classmethod
    def run_causality_ablation(
        cls,
        dataset_name: str = "mc_maze",
        seeds: Tuple[int, ...] = (42, 43, 44),
        num_steps: int = 20,
    ) -> List[AblationResult]:
        """Family 4: Causal temporal attention vs bidirectional full attention."""
        variants = ["causal_temporal", "bidirectional_full"]
        results = []

        for variant in variants:
            for s in seeds:
                cfg = AblationConfig(
                    family=AblationFamily.TEMPORAL_CAUSALITY,
                    variant_name=variant,
                    dataset_name=dataset_name,
                    seed=s,
                    num_steps=num_steps,
                )
                res = cls._run_single_ablation(cfg)
                results.append(res)
        return results

    @classmethod
    def run_dynamics_ablation(
        cls,
        dataset_name: str = "linear_circuit",
        seeds: Tuple[int, ...] = (42, 43, 44),
        num_steps: int = 20,
    ) -> List[AblationResult]:
        """Family 5: Autonomous continuous flow vs inferred external drive."""
        variants = ["autonomous_flow", "inferred_drive"]
        results = []

        for variant in variants:
            for s in seeds:
                cfg = AblationConfig(
                    family=AblationFamily.DYNAMICS_DRIVE,
                    variant_name=variant,
                    dataset_name=dataset_name,
                    seed=s,
                    num_steps=num_steps,
                )
                res = cls._run_single_ablation(cfg)
                results.append(res)
        return results

    @classmethod
    def _run_single_ablation(cls, config: AblationConfig) -> AblationResult:
        """Execute a single ablation experiment configuration."""
        t0 = time.perf_counter()
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        train_arr, test_arr, held_in, held_out, _ = BenchmarkSuiteRunner.load_dataset(
            config.dataset_name, seed=config.seed
        )
        K_tr, T, N = train_arr.shape
        K_te, _, _ = test_arr.shape
        dt = 0.02

        null_rate_marginal = np.maximum(np.mean(train_arr, axis=(0, 1), keepdims=True) / dt, 1e-4)
        null_rates = np.tile(null_rate_marginal, (K_te, T, 1))

        bps_eval = BitsPerSpikeEvaluator(bin_width_s=dt)
        dev_calc = EvaluationMetricsCalculator(bin_width_s=dt)

        # -------------------------------------------------------------
        # Family 5: VLDM Dynamics Drive Ablation
        # -------------------------------------------------------------
        if config.family == AblationFamily.DYNAMICS_DRIVE:
            use_drive = (config.variant_name == "inferred_drive")
            cfg = VLDMConfig(
                obs_dim=N,
                latent_dim=4,
                encoder_hidden_dim=32,
                decoder_hidden_dim=32,
                bin_width_s=dt,
                use_inferred_drive=use_drive,
                drive_dim=2,
            )
            model = VariationalLatentDynamicsModel(cfg)
            opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

            x_tr = torch.from_numpy(train_arr).float()

            model.train()
            for _ in range(config.num_steps):
                opt.zero_grad()
                rates, _, loss, _ = model(x_tr)
                loss.backward()
                opt.step()

            model.eval()
            x_te = torch.from_numpy(test_arr).float()
            with torch.no_grad():
                pred_rates_t, _, _, _ = model(x_te)
                pred_rates = pred_rates_t.cpu().numpy()

        # -------------------------------------------------------------
        # Families 1 to 4: Spatiotemporal Attention Model Ablations
        # -------------------------------------------------------------
        else:
            # Configure architecture based on family variant
            d_model = 32
            num_layers = 2
            causal = False

            if config.family == AblationFamily.DEPTH_CAPACITY:
                if config.variant_name == "depth1_dim32":
                    num_layers, d_model = 1, 32
                elif config.variant_name == "depth2_dim64":
                    num_layers, d_model = 2, 64
                elif config.variant_name == "depth4_dim64":
                    num_layers, d_model = 4, 64

            if config.family == AblationFamily.TEMPORAL_CAUSALITY:
                causal = (config.variant_name == "causal_temporal")

            att_cfg = AttentionConfig(
                obs_dim=N,
                d_model=d_model,
                num_heads=2,
                num_layers=num_layers,
                bin_width_s=dt,
                causal=causal,
            )
            model = SpatiotemporalAttentionModel(att_cfg, latent_dim=4)
            opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

            x_tr = torch.from_numpy(train_arr).float()
            x_te = torch.from_numpy(test_arr).float()

            # Determine observation masks
            obs_mask = torch.ones((K_tr, T, N), dtype=torch.bool)
            if config.family == AblationFamily.INPUT_MASKING:
                if config.variant_name == "co_smoothing":
                    obs_mask[:, :, held_out] = False
                elif config.variant_name == "unmasked":
                    pass  # all True
                elif "token_dropout" in config.variant_name:
                    p_drop = 0.2
                    rand_drop = torch.rand((K_tr, T, N)) < p_drop
                    obs_mask[rand_drop] = False
            else:
                # Default co-smoothing mask
                obs_mask[:, :, held_out] = False

            # Custom objective loss selection (Family 1)
            model.train()
            loss_poisson = PoissonLoss(bin_width_s=dt)

            for _ in range(config.num_steps):
                opt.zero_grad()
                rates, _, default_loss, _ = model(x_tr, observed_mask=obs_mask, target_counts=x_tr)

                if config.family == AblationFamily.OBJECTIVE and config.variant_name == "mse":
                    # Unweighted mean squared error in count space
                    counts_pred = rates * dt
                    loss = F.mse_loss(counts_pred, x_tr)
                elif config.family == AblationFamily.OBJECTIVE and config.variant_name == "gaussian":
                    # Gaussian NLL with unit variance
                    counts_pred = rates * dt
                    loss = 0.5 * F.mse_loss(counts_pred, x_tr) + 0.5 * np.log(2 * np.pi)
                else:
                    loss = default_loss

                loss.backward()
                opt.step()

            model.eval()
            obs_mask_te = torch.ones((K_te, T, N), dtype=torch.bool)
            if config.family == AblationFamily.INPUT_MASKING and config.variant_name == "unmasked":
                pass
            else:
                obs_mask_te[:, :, held_out] = False

            with torch.no_grad():
                pred_rates_t, _, _, _ = model(x_te, observed_mask=obs_mask_te)
                pred_rates = pred_rates_t.cpu().numpy()

        res_held_in = bps_eval.evaluate(
            predicted_rates_hz=pred_rates[:, :, held_in],
            target_counts=test_arr[:, :, held_in],
            null_rates_hz=null_rates[:, :, held_in],
        )
        res_held_out = bps_eval.evaluate(
            predicted_rates_hz=pred_rates[:, :, held_out],
            target_counts=test_arr[:, :, held_out],
            null_rates_hz=null_rates[:, :, held_out],
        )
        metrics = dev_calc.compute_metrics(
            predicted_rates_hz=pred_rates,
            target_counts=test_arr,
            null_rates_hz=null_rates,
        )

        dur = time.perf_counter() - t0
        return AblationResult(
            config=config,
            bps_held_in=res_held_in.population_bps,
            bps_held_out=res_held_out.population_bps,
            poisson_deviance=metrics.poisson_deviance,
            pseudo_r2=metrics.deviance_pseudo_r2 if metrics.deviance_pseudo_r2 is not None else 0.0,
            duration_s=dur,
        )
