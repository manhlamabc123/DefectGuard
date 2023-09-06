#!/bin/bash

torch-model-archiver --model-name cc2vec \
    --version 1.0 \
    --model-file model.py \
    --serialized-file cc2vec.pt \
    --handler handler.py \
    --extra-files cc2vec.json,qt_dict.pkl,deepjit_extended.pt

mv cc2vec.mar ../torchserve/model-store