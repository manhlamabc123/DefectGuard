#!/bin/bash

torch-model-archiver --model-name deepjit \
    --version 1.0 \
    --model-file model.py \
    --serialized-file deepjit.pt \
    --handler handler.py \
    --extra-files deepjit.json,platform_dict.pkl

mv deepjit.mar ../torchserve/model-store