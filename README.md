# Hybrid SRL Replay Buffer for Inference-Time Self-Training in LLMs

## Overview
This repository (or code snippet) contains a Python implementation of a **Hybrid Spaced Repetition Learning (SRL) Replay Buffer** integrated with diversity bonuses from the Exploratory Iteration (ExIt) framework, as described in the paper "Bootstrapping Task Spaces for Self-Improvement" (arXiv:2509.04575). The buffer is designed for use in systems where large language models (LLMs) perform self-training and fine-tuning primarily at inference time, modeled as Partially Observable Markov Decision Processes (POMDPs).

The code provides a class `HybridSRLReplayBuffer` that stores experiences (e.g., POMDP trajectories like state-action-observation-reward sequences) and manages their replay using SRL principles (adaptive intervals based on recall success) weighted by ExIt-inspired diversity metrics (reward variance and embedding distances). This hybrid approach aims to balance knowledge retention (via SRL) with exploration of underrepresented task paths (via diversity bonuses), enabling more robust, autonomous self-improvement in LLMs.

## Conversation Background
Our discussion began with an explanation of the arXiv paper on bootstrapping task spaces for LLM self-improvement using ExIt and Group-Relative Policy Optimization (GRPO). We explored analogies between in-context learning (ICL) and inference-time optimization, then delved into designing systems for executing POMDPs at inference time. This evolved into ideas for shifting the bulk of fine-tuning to runtime via self-training loops, incorporating replay buffers to stabilize updates and prevent catastrophic forgetting.

Key evolution:
- **POMDP at Inference**: Discussed online solvers like POMCP for runtime belief updates and planning.
- **Inference-Time Fine-Tuning**: Proposed computing gradients during inference and integrating them periodically, inspired by Test-Time Training (TTT) and self-training with pseudo-labels.
- **Replay Feature**: Debated the need for a replay buffer; concluded it's essential for stability in continual learning, especially in POMDPs.
- **SRL Integration**: Suggested adapting spaced repetition (from educational tools like Anki) to the buffer for efficient, human-like retention—short intervals for hard items, longer for mastered ones.
- **Hybrid with ExIt/GRPO**: Enhanced SRL by weighting intervals with ExIt's diversity bonuses (variance and embedding distances) to encourage exploring novel paths, preventing mode collapse.
- **Simulation and Code**: Iterated on pseudocode, simulated with dummy data, fixed issues (e.g., sorting for harder items first), and refined into a working implementation.

The conversation was iterative, with simulations to validate the hybrid mechanics, ensuring high-diversity experiences get shorter replay intervals for better exploration.

## Why We Created This Code
The primary motivation was to prototype a core component for a self-improving AI system that aligns with the paper's ExIt framework but extends it to inference-time adaptation. Traditional fine-tuning is offline and resource-intensive; shifting it to runtime allows for continual, personalized learning from real interactions (e.g., user queries or task refinements).

Specific reasons:
- **Addressing Limitations in the Paper**: ExIt uses a task buffer for autocurricula, but lacks explicit mechanisms for long-term retention in continual settings. SRL adds this, while diversity weighting preserves ExIt's exploration focus.
- **Enabling POMDP-Based Self-Training**: In POMDPs (as modeled in the paper for multi-step self-improvement), partial observability requires replaying diverse trajectories to refine beliefs. This buffer supports that at inference without full retraining.
- **Practical Exploration**: We wanted a testable artifact to simulate how such a buffer behaves—e.g., prioritizing high-variance/diversity items for GRPO updates, mimicking human learning (spaced repetition) in AI agents.
- **Broader Implications**: This could enhance agentic LLMs (e.g., for math solving, tool-use, or coding as in the paper's experiments) by making them more adaptive, efficient, and robust to data shifts during deployment.

The code was built incrementally through discussion, with simulations to debug and demonstrate dynamics like interval shortening for diverse, hard experiences.

## What We Have Left to Do
While the buffer is functional in isolation, integrating it into a full system remains. Potential next steps include:

1. **Integration with LLMs and Fine-Tuning Loops**:
   - Embed this buffer in a PyTorch-based training loop (e.g., using Hugging Face Transformers).
   - Compute real embeddings from LLM hidden states (e.g., via Sentence-BERT or model outputs) instead of dummies.
   - Hook into GRPO or PPO for policy updates, using replayed batches to calculate advantages.

2. **POMDP and Task-Specific Testing**:
   - Test on paper-inspired domains: MATH dataset for math refinement, ToolBench for multi-turn tool-use, ML-Bench for coding.
   - Simulate POMDPs (e.g., using libraries like POMDPs.jl or custom grid-worlds) to generate trajectories for the buffer.
   - Evaluate metrics: Improvement in solution quality over iterations, resistance to forgetting, diversity in generated paths.

3. **Enhancements to the Buffer**:
   - Add ε-greedy exploration: Flag new experiences as "divergence" to boost initial diversity.
   - Improve Eviction: Prioritize removing low-diversity or well-mastered items (beyond FIFO).
   - Handle Scalability: Cluster experiences for large buffers (e.g., via scikit-learn K-means) or use approximate nearest neighbors (FAISS) for distances.
   - Incorporate Uncertainty: Weight by Bayesian estimates for noisy recalls in sparse-reward POMDPs.

4. **Real-World Deployment Considerations**:
   - Optimize for Latency: Asynchronous updates to avoid slowing inference.
   - Privacy/Safety: Ensure self-generated data doesn't amplify biases; add filters.
   - Benchmarks: Compare against baselines like standard replay or pure ExIt on continual learning tasks (e.g., CIFAR-100 splits).

5. **Research and Validation**:
   - Draw from related papers (e.g., TFC-SR for SRL in RL, or recent TTT works).
   - Run ablation studies: SRL alone vs. hybrid vs. no replay.
   - Scale to Larger Models: Test with small LLMs (e.g., GPT-2) before massive ones.

This is a starting point work to do to expand it into a full experiment


## Dependencies
- NumPy (for embeddings and norms)

## License
MIT License 
