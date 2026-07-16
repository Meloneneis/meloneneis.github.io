---
title: "Deep Q-Network Racing Agent"
excerpt: "Vision-based DQN with configurable CNN/FC stacks, optional dueling heads, mixed-precision benchmarks, and Weights & Biases tracking."
collection: portfolio
permalink: /portfolio/dqn-racing/
---

## Problem
Control from pixels is brittle: architecture choices, precision modes, and action spaces interact. Racing environments amplify any instability in the learning loop.

## Approach
A from-scratch PyTorch DQN stack with:

- Configurable convolutional and fully-connected widths
- Optional **dueling** value/advantage heads
- Replay buffer, schedules, and evaluation harnesses
- GPU mixed-precision benchmarks and multi-run checkpoints (`chocolate`, `fiery`, `gallant`, …)
- Experiment metadata logged through Weights & Biases

## Signal
Demonstrates systems-level RL hygiene: measurable training, ablation-friendly model construction, and hardware-aware iteration — not a single notebook demo.

## Links
- Repository: [DQN_Training](https://github.com/Meloneneis/DQN_Training)
