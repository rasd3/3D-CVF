# 3D-CVF
This is the official implementation of [3D-CVF: Generating Joint Camera and LiDAR Features Using Cross-View Spatial Feature Fusion for 3D Object Detection](https://arxiv.org/abs/2004.12636), built on [SECOND](https://github.com/traveller59/second.pytorch).

## Requirements

Follow the installation steps in [SECOND](https://github.com/traveller59/second.pytorch), or use the docker image we provide.

```
docker pull yckimm/second:second_v1.5
```

## Getting Started

### Training (1st stage)

```
sh train_bash_1st.sh
```

### Training (2nd stage)

```
sh train_bash_2nd.sh
```



## Acknowledge

Thanks to the [SECOND](https://github.com/traveller59/second.pytorch) codebase maintained by traveller59.