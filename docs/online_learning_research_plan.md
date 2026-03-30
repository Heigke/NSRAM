# Online Learning Research Plan — Beyond NS-RAM

## The Gap Nobody Has Filled

The pieces exist. Nobody has assembled them:

| Component | Source | What it does | Scale tested |
|-----------|--------|-------------|-------------|
| Surprise-gated memory | Titans (Google, Jan 2025) | Memorize unexpected, forget routine | 760M |
| Multi-timescale updates | Hope/Nested Learning (Google, NeurIPS 2025) | Different update frequencies per module | 1.3B |
| Sparse subspace isolation | Adaptive SVD (Red Hat, Apr 2025) | Update orthogonal to critical knowledge | 7B |
| Self-distillation anchor | SDFT (Jan 2026) | Model teaches itself to retain knowledge | 14B |
| MoE + LoRA routing | D-MoLE / SMoLoRA (2025-2026) | Per-task experts with soft routing | 13B |
| Plasticity injection | AdaLin (May 2025) | Prevent dead neurons in continual learning | ResNet-18 |
| Physics-based consolidation | Our FIOT (today) | Fisher→oxide trapping, 69pp→22pp forgetting | MNIST |

## Novel Combination: SOLE (Surprise-Orthogonal Learning with self-distillation and Experts)

```
For each input:
  1. SURPRISE: Is this input surprising? (Titans-style prediction error)
     - High surprise → learn aggressively (new knowledge)
     - Low surprise → skip update (already known)

  2. SUBSPACE: Where to update? (SVD-style orthogonal projection)
     - Decompose weight gradient via SVD
     - Project update orthogonal to high-Fisher directions (critical knowledge)
     - Only modify the "free" subspace

  3. EXPERT: Which expert handles this? (MoE routing)
     - Route to task-appropriate expert
     - Spawn new expert if no existing one fits
     - Old experts: frozen or slow-updating

  4. DISTILL: Don't forget yourself (SDFT-style)
     - Before updating, generate predictions on held-out data
     - After updating, minimize divergence from pre-update predictions
     - This is "self-replay" without storing data

  5. CONSOLIDATE: Fast→slow (Hope-style multi-timescale)
     - Fast weights: updated every step (high plasticity)
     - Slow weights: updated every N steps (high stability)
     - Consolidation: EMA of fast→slow when surprise is low
```

## Implementation Plan

### Phase 1: Baseline reproduction (GPU, standard PyTorch)

**File**: `examples/sole_experiment.py`

Reproduce on MNIST → Fashion-MNIST → KMNIST (3 sequential tasks):
- Finetune baseline (catastrophic forgetting)
- EWC baseline (our proven 0.1pp result)
- Adaptive SVD baseline (literature: best practical)
- LoRA per-task baseline

### Phase 2: SOLE prototype

**Components to implement:**
1. Surprise gate: prediction error on current batch vs model expectation
2. SVD orthogonal projection: using `torch.linalg.svd` on gradient
3. Simple MoE: 2-4 experts with softmax router
4. Self-distillation: cache logits before update, KL-div after
5. Fast/slow EMA: two weight copies, periodic consolidation

**Architecture**: Small MLP (784→500→500→10) — fast iteration.
Then scale to ViT-tiny or small transformer if it works.

### Phase 3: Scaling test

If Phase 2 shows promise on MNIST-scale:
- Test on Split-CIFAR-100 (20 sequential tasks)
- Test on a small LM task (sequential domain adaptation)
- Compare to D-MoLE numbers (BWT = -1.49%)

### Phase 4: NS-RAM mapping

Map each SOLE component to NS-RAM physics:
- Surprise gate → avalanche threshold (high input = above BVpar)
- SVD projection → modular NS-RAM sub-arrays
- Fast/slow → body potential / oxide trapping
- Self-distillation → periodic readout + replay cycle

## Key Questions to Answer

1. Does surprise gating actually help? (Titans claims yes, nobody has tested for CL)
2. Does SVD projection + self-distillation give better results than EWC alone?
3. How many experts are needed for 5/10/20 sequential tasks?
4. What is the compute overhead of SVD projection per step?
5. Does the combination outperform any individual component?

## Success Metrics

| Metric | Finetune | EWC | Target (SOLE) |
|--------|----------|-----|---------------|
| MNIST→FMNIST forgetting | 69pp | 0.1pp | <1pp |
| MNIST→FMNIST→KMNIST joint | ~40% | ~85% | >88% |
| Compute overhead vs finetune | 1x | 1.5x | <3x |
| Memory overhead | 1x | 2x | <4x |
