#!/bin/bash


if [ "$USER" == "chenpansong" ] || [ "$USER" == "linshenghao" ]; then
    COMPOSE_FILE="carla-compose-${USER}.yml"
else
    COMPOSE_FILE="carla-compose.yml"
fi

echo "${COMPOSE_FILE}"

docker-compose -f $COMPOSE_FILE  -p carla-${USER}  up -d
