#!/bin/bash

wget https://s3.eu-central-1.amazonaws.com/corupublic/mtlcc/pytorch/data.zip
unzip data.zip
rm data.zip

mkdir -p checkpoints
cd checkpoints
wget https://s3.eu-central-1.amazonaws.com/corupublic/mtlcc/pytorch/model_00.pth
