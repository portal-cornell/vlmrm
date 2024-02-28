# Navigating the repo
1. 


# Humanoid training detal
For all humanoid experiments, we use SAC with the same set of hyperparameters tuned on prelimi-
nary experiments with the kneeling task. We train for 10 million steps with an episode length of 100
steps. Learning starts after 50000 initial steps and we do 100 SAC updates every 100 environment
steps. We use SAC parameters τ = 0.005, γ = 0.95, and learning rate 6 · 10−4. We save a model
checkpoint every 128000 steps. For our final evaluation, we always evaluate the checkpoint with the
highest training reward. We parallelize rendering over 4 GPUs, and also use batch size B = 3200
for evaluating the CLIP rewards.