#!/bin/bash

VOLUMES="--volume=$XSOCK:$XSOCK:rw \
         --volume=$XAUTH:$XAUTH:rw \
         --volume=/home/linshenghao/carla_data/:/home/carla/.config/Epic/CarlaUE4/Saved:rw"

docker run --name="carla-$USER" \
  -d --rm \
  -p 5000-5002:5000-5002 \
  $VOLUMES\
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --gpus 'device=1' \
  carlasim/carla:0.9.10.1 \
  /bin/bash -c  \
  'SDL_VIDEODRIVER=offscreen CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=1 ./CarlaUE4.sh -ResX=640 -ResY=360 \
  -nosound -windowed -opengl \
  -carla-rpc-port=5000 \
  -quality-level=Epic'

