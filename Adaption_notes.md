# Commands to run

## Training
`vlmrm train "$(cat <path-to-the-config-yaml-file>.yaml)"`

For example, `vlmrm train "$(cat configs/standup_config.yaml)"`

## Inference
`vlmrm generate_dataset "$(cat <path-to-the-inference-config-yaml-file>.yaml)"`

For example, `vlmrm generate_dataset "$(cat configs/inference.yaml)"`


# Navigating the repo
under src/vlmrm
- cli: define `vlmrm train` and `vlmrm generate_dataset` which is used for inference 
- contrib/sb3: is where the RL training agent is defined


# Configuration Files
* **config.yaml:** used to train an agent in the environment to stand up
* **gt_config.yaml:** used to train the agent with the environment's ground truth reward (in humanoid standup environment)
* **original_config:** used to trian the agent in the original humanoid environment (when the humanoid is spawn standing upright)


### Camera Configuration
For humanoid tasks, they have a custom camera angle to improve performance. They didn't provide the code/value, so these are the approximate values
```
camera_config:
    lookat: [0.25, 0, 1.25]  # x, y, z
    distance: 3.5  # Distance from the camera to the humanoid
    azimuth: 180  # Make camera look at negative x (90 = positive y, 0 = positive x)
    elevation: -10  # How high the camera is above ground
```