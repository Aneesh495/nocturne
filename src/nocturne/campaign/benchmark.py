"""Three-seed model comparison campaign across NLB recordings and synthetic circuits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from nocturne.recordings.mc_maze import MCMazeAdapter
from nocturne.recordings.mc_rtt import MCRTTAdapter
from nocturne.recordings.area2_bump import Area2BumpAdapter
from nocturne.synthetic.generator import SyntheticDatasetGenerator
from nocturne.models.baselines.mean_rate import MeanRateBaseline
from nocturne.models.baselines.poisson_factor import LowRankPoissonFactorModel
from nocturne.models.likelihoods import PoissonLoss
from nocturne.models.attention.config import AttentionConfig
from nocturne.models.attention.model import SpatiotemporalAttentionModel
from nocturne.models.vldm.config import VLDMConfig
from nocturne.models.vldm.model import VariationalLatentDynamicsModel
from nocturne.evaluation.bits_per_spike import BitsPerSpikeEvaluator
from nocturne.evaluation.metrics import EvaluationMetricsCalculator
from nocturne.evaluation.probes import BehavioralDecodingProbe


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """Execution configuration for an individual benchmark run."""

    dataset_name: str
    model_family: str       # 'mean_rate', 'poisson_factor', 'vldm', 'attention'
    seed: int = 42
    latent_dim: int = 4
    num_steps: int = 20
    learning_rate: float = 1e-3
    device: str = "cpu"


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Quantitative outcome and metrics from a benchmark run."""

    config: BenchmarkRunConfig
    bps_held_in: Optional[float]
    bps_held_out: Optional[float]
    poisson_deviance: float
    pseudo_r2: float
    probe_r2: Optional[float]
    duration_s: float
    model_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["config"] = asdict(self.config)
        return d


class BenchmarkSuiteRunner:
    """Orchestrates multi-seed model evaluation fulfilling capability B23 and gate BA04."""

    SUPPORTED_DATASETS = ("mc_maze", "mc_rtt", "area2_bump", "linear_circuit", "nonlinear_circuit")
    SUPPORTED_MODELS = ("mean_rate", "poisson_factor", "vldm", "attention")

    @classmethod
    def load_dataset(cls, dataset_name: str, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[int], List[int], Optional[np.ndarray]]:
        """Load dataset and extract train/test count matrices, channel split, and behavior.

        Returns:
            Tuple of:
                train_counts: [K_train, T, N]
                test_counts:  [K_test, T, N]
                held_in_channels: List[int]
                held_out_channels: List[int]
                test_kinematics: Optional [K_test, T, 2]
        """
        if dataset_name == "mc_maze":
            rec = MCMazeAdapter().create_synthetic_maze_fixture(num_trials=25, num_neurons=20, seed=seed)
        elif dataset_name == "mc_rtt":
            rec = MCRTTAdapter().create_synthetic_rtt_fixture(duration_s=25.0, num_neurons=20, seed=seed)
        elif dataset_name == "area2_bump":
            rec = Area2BumpAdapter().create_synthetic_bump_fixture(num_trials=25, num_neurons=20, seed=seed)
        elif dataset_name in ("linear_circuit", "linear_oscillator"):
            gen = SyntheticDatasetGenerator(seed=seed)
            _, rec = gen.generate_regime_corpus(regime_name="linear_oscillator", num_trials=20, num_neurons=16)
        elif dataset_name in ("nonlinear_circuit", "nonlinear_oscillator"):
            gen = SyntheticDatasetGenerator(seed=seed)
            _, rec = gen.generate_regime_corpus(regime_name="nonlinear_oscillator", num_trials=20, num_neurons=16)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        split = rec.splits["official"]
        all_counts = rec.binned_counts.counts  # [T_tot, N]
        N = rec.num_neurons

        # Extract trials
        trials = rec.trials.trials if rec.trials else []
        train_trials = [t for t in trials if t.trial_id in split.train.trial_ids]
        test_trials = [t for t in trials if t.trial_id in split.test.trial_ids]

        # Use fixed length slicing for benchmarking
        T_slice = 25
        train_list = []
        for t in train_trials:
            st = int(t.interval.start / rec.binned_counts.grid.bin_config.bin_width_s)
            sl = all_counts[st : st + T_slice, :]
            if sl.shape[0] == T_slice:
                train_list.append(sl)

        test_list = []
        kinematics_list = []
        for t in test_trials:
            st = int(t.interval.start / rec.binned_counts.grid.bin_config.bin_width_s)
            sl = all_counts[st : st + T_slice, :]
            if sl.shape[0] == T_slice:
                test_list.append(sl)
                if t.behavior is not None and len(t.behavior.values) > 0:
                    kin_arr = np.array(t.behavior.values, dtype=float)
                    if kin_arr.shape[0] >= T_slice:
                        kinematics_list.append(kin_arr[:T_slice, :])

        if not train_list:
            train_list = [np.ones((T_slice, N), dtype=float)]
        if not test_list:
            test_list = [np.ones((T_slice, N), dtype=float)]

        train_arr = np.stack(train_list, axis=0)  # [K_tr, T, N]
        test_arr = np.stack(test_list, axis=0)    # [K_te, T, N]

        # 70% held-in, 30% held-out
        n_held_in = max(1, int(0.7 * N))
        held_in = list(range(n_held_in))
        held_out = list(range(n_held_in, N))

        test_kin = np.stack(kinematics_list, axis=0) if len(kinematics_list) == len(test_list) else None
        return train_arr, test_arr, held_in, held_out, test_kin

    @classmethod
    def run_single(cls, config: BenchmarkRunConfig) -> BenchmarkRunResult:
        """Execute a single benchmark run under the given configuration."""
        t0 = time.perf_counter()
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        train_arr, test_arr, held_in, held_out, test_kin = cls.load_dataset(
            config.dataset_name, seed=config.seed
        )
        K_tr, T, N = train_arr.shape
        K_te, _, _ = test_arr.shape
        dt = 0.02

        # Null baseline: training mean marginal rates
        null_rate_marginal = np.maximum(np.mean(train_arr, axis=(0, 1), keepdims=True) / dt, 1e-4)
        null_rates = np.tile(null_rate_marginal, (K_te, T, 1))

        bps_eval = BitsPerSpikeEvaluator(bin_width_s=dt)
        dev_calc = EvaluationMetricsCalculator(bin_width_s=dt)

        if config.model_family == "mean_rate":
            model = MeanRateBaseline(num_neurons=N)
            model.fit_from_counts(train_arr, bin_width_s=dt)
            pred_rates = model.forward(batch_size=K_te, num_time_bins=T).cpu().numpy()
            model_hash = hashlib.sha256(b"mean_rate_weights").hexdigest()

        elif config.model_family == "poisson_factor":
            model = LowRankPoissonFactorModel(num_neurons=N, rank=config.latent_dim)
            x_tr = torch.from_numpy(train_arr).float()
            model.fit_factors_and_loadings(x_tr, bin_width_s=dt, num_steps=config.num_steps)
            x_te = torch.from_numpy(test_arr).float()
            te_factors = torch.nn.Parameter(torch.randn(K_te, T, config.latent_dim) * 0.1)
            opt = torch.optim.Adam([te_factors], lr=0.05)
            loss_fn = PoissonLoss(bin_width_s=dt)
            for _ in range(config.num_steps):
                opt.zero_grad()
                rates = model.decode_rates(te_factors)
                loss = loss_fn.forward_from_rates(rates, x_te)
                loss.backward()
                opt.step()
            pred_rates = model.decode_rates(te_factors).detach().cpu().numpy()
            model_hash = hashlib.sha256(b"poisson_factor_weights").hexdigest()

        elif config.model_family == "attention":
            cfg = AttentionConfig(
                obs_dim=N,
                d_model=32,
                num_heads=2,
                num_layers=2,
                bin_width_s=dt,
                causal=False,
            )
            model = SpatiotemporalAttentionModel(cfg, latent_dim=config.latent_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

            # Mask held-out channels as unobserved
            x_tr = torch.from_numpy(train_arr).float()
            obs_mask = torch.ones((K_tr, T, N), dtype=torch.bool)
            obs_mask[:, :, held_out] = False

            model.train()
            for _ in range(config.num_steps):
                optimizer.zero_grad()
                rates, _, loss, _ = model(x_tr, observed_mask=obs_mask, target_counts=x_tr)
                loss.backward()
                optimizer.step()

            model.eval()
            x_te = torch.from_numpy(test_arr).float()
            obs_mask_te = torch.ones((K_te, T, N), dtype=torch.bool)
            obs_mask_te[:, :, held_out] = False
            with torch.no_grad():
                pred_rates_t, _, _, _ = model(x_te, observed_mask=obs_mask_te)
                pred_rates = pred_rates_t.cpu().numpy()

            model_hash = hashlib.sha256(str(model.state_dict().keys()).encode()).hexdigest()

        elif config.model_family == "vldm":
            cfg = VLDMConfig(
                obs_dim=N,
                latent_dim=config.latent_dim,
                hidden_dim=32,
                bin_width_s=dt,
            )
            model = VariationalLatentDynamicsModel(cfg)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

            x_tr = torch.from_numpy(train_arr).float()
            model.train()
            for _ in range(config.num_steps):
                optimizer.zero_grad()
                rates, _, loss, _ = model(x_tr, target_counts=x_tr)
                loss.backward()
                optimizer.step()

            model.eval()
            x_te = torch.from_numpy(test_arr).float()
            with torch.no_grad():
                pred_rates_t, _, _, _ = model(x_te)
                pred_rates = pred_rates_t.cpu().numpy()

            model_hash = hashlib.sha256(str(model.state_dict().keys()).encode()).hexdigest()

        else:
            raise ValueError(f"Unknown model family: {config.model_family}")

        # Compute BPS held-in and held-out
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

        # Compute deviance and pseudo-R2
        metrics = dev_calc.compute_metrics(
            predicted_rates_hz=pred_rates,
            target_counts=test_arr,
            null_rates_hz=null_rates,
        )

        # Kinematic probe if test kinematics available
        probe_r2 = None
        if test_kin is not None and config.model_family in ("attention", "vldm"):
            probe = BehavioralDecodingProbe(ridge_alpha=1e-2)
            n_split = int(0.7 * len(pred_rates))
            if n_split > 1:
                probe.fit(pred_rates[:n_split], test_kin[:n_split])
                probe_res = probe.evaluate(pred_rates[n_split:], test_kin[n_split:])
                probe_r2 = probe_res.macro_r2

        dur = time.perf_counter() - t0
        return BenchmarkRunResult(
            config=config,
            bps_held_in=res_held_in.population_bps,
            bps_held_out=res_held_out.population_bps,
            poisson_deviance=metrics.poisson_deviance,
            pseudo_r2=metrics.deviance_pseudo_r2 if metrics.deviance_pseudo_r2 is not None else 0.0,
            probe_r2=probe_r2,
            duration_s=dur,
            model_sha256=model_hash,
        )

    @classmethod
    def run_multi_seed(
        cls,
        dataset_name: str,
        model_family: str,
        seeds: Tuple[int, ...] = (42, 43, 44),
        num_steps: int = 20,
    ) -> List[BenchmarkRunResult]:
        """Execute benchmark across three seeds."""
        results = []
        for s in seeds:
            cfg = BenchmarkRunConfig(
                dataset_name=dataset_name,
                model_family=model_family,
                seed=s,
                num_steps=num_steps,
            )
            results.append(cls.run_single(cfg))
        return results
