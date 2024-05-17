# Dataset Cards and Public Electrophysiology Provenance

This document specifies the experimental origins, subject populations, surgical implant targets, task paradigms, trial alignments, and split definitions for all primary electrophysiology recordings evaluated in Nocturne.

---

## 1. MC_Maze (Primary Motor & Dorsal Premotor Cortex)

- **Archive / Source**: DANDI Archive (Dandiset `000128`), Neural Latents Benchmark `mc_maze`.
- **Primary Investigator / Lab**: Churchland Lab, Kaufman et al. (2014, 2016).
- **Animal Subject**: Adult male rhesus macaque (*Macaca mulatta*, Subject "Jenkins", age ~8.0 years).
- **Implanted Structures**:
  - Primary motor cortex (M1, caudal array).
  - Dorsal premotor cortex (PMd, rostral array).
  - Dual 96-channel planar Blackrock silicon microelectrode arrays (Utah arrays, 1.5 mm electrode length).
- **Experimental Paradigm**:
  - Delayed-reach center-out obstacle avoidance task ("Maze").
  - Subject guides a virtual fingertip cursor on a 2D screen using manipulandum position.
  - Barriers require curved reach trajectories across 108 distinct maze configurations.
- **Trial Structure and Alignment**:
  - Trial phases: Target presentation $\to$ Variable delay epoch (450–950 ms) $\to$ Go cue $\to$ Movement onset $\to$ Target acquisition $\to$ Reward.
  - Standard evaluation alignment: $250\text{ ms}$ prior to movement onset to $450\text{ ms}$ post-movement onset ($700\text{ ms}$ evaluation window).
  - Binning: $\Delta t = 20\text{ ms}$ ($35$ time bins) or $\Delta t = 5\text{ ms}$ ($140$ time bins).
- **Observation Roles**:
  - Total sorted units: ~138–182 units (depending on session quality thresholds).
  - Co-smoothing partition: ~70% designated as `HELD_IN`, ~30% designated as `HELD_OUT`.
- **Split Structure**:
  - Official NLB split: Tripartite trial partition ($~60\%$ train, $~20\%$ validation, $~20\%$ local held-out test).
  - Chronological stress split: Strict temporal ordering across session elapsed time.
- **License / Terms**: Creative Commons Attribution 4.0 International (CC-BY 4.0).

---

## 2. MC_RTT (Primary Motor Cortex Continuous Tracking)

- **Archive / Source**: DANDI Archive (Dandiset `000129`), Neural Latents Benchmark `mc_rtt`.
- **Primary Investigator / Lab**: Carmena Lab, O'Doherty et al. (2017, 2018).
- **Animal Subject**: Adult male rhesus macaque (*Macaca mulatta*, Subject "Indy", age ~7.5 years).
- **Implanted Structures**:
  - Primary motor cortex (M1, arm representation).
  - Single 96-channel Blackrock microelectrode array.
- **Experimental Paradigm**:
  - Continuous random target pursuit (Random Target Tracking, RTT).
  - Continuous pursuit task without discrete delayed reach trials.
  - Targets appear at pseudo-random positions upon acquisition of the prior target.
- **Trial Structure and Slicing**:
  - Non-trial structured continuous electrophysiological recording.
  - Nocturne applies `IntervalCutter` to partition uninterrupted epochs into $1000\text{ ms}$ ($50$ bins at $\Delta t = 20\text{ ms}$) or $600\text{ ms}$ non-overlapping segments.
  - Gaps between discontinuous recording blocks are strictly quarantined via `GapDetector`.
- **Observation Roles**:
  - Total sorted units: ~130 units.
  - Co-smoothing partition: 70% held-in ($~91$ units), 30% held-out ($~39$ units).
- **Split Structure**:
  - Chronological block split: First $70\%$ of elapsed continuous time to train, next $15\%$ to validation, final $15\%$ to held-out test.
- **License / Terms**: CC-BY 4.0.

---

## 3. Area2_Bump (Primary Somatosensory Cortex Mechanical Perturbations)

- **Archive / Source**: DANDI Archive (Dandiset `000127`), Neural Latents Benchmark `area2_bump`.
- **Primary Investigator / Lab**: Miller Lab, Chowdhury et al. (2020).
- **Animal Subject**: Adult male rhesus macaque (*Macaca mulatta*, Subject "Han", age ~9.0 years).
- **Implanted Structures**:
  - Brodmann Area 2 of primary somatosensory cortex (S1, limb proprioception).
  - 96-channel Utah microelectrode array.
- **Experimental Paradigm**:
  - Center-out reaching with passive mechanical limb perturbations ("bumps").
  - A robotic manipulandum applies an unexpected brief mechanical force pulse (duration $125\text{ ms}$) displacing the arm in 1 of 4 or 8 radial directions.
- **Trial Structure and Alignment**:
  - Aligned to `bump_onset`: $150\text{ ms}$ pre-bump to $350\text{ ms}$ post-bump ($500\text{ ms}$ evaluation window).
  - Captures high-gain phasic proprioceptive feedback and voluntary feedback corrections.
  - Binning: $\Delta t = 20\text{ ms}$ ($25$ bins) or $\Delta t = 5\text{ ms}$ ($100$ bins).
- **Observation Roles**:
  - Total sorted units: ~50–70 units.
  - Co-smoothing partition: 70% held-in, 30% held-out.
- **Split Structure**:
  - Official condition-stratified split ensuring identical representation of perturbation angles across train, validation, and test.
- **License / Terms**: CC-BY 4.0.
