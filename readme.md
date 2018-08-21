# Multitemporal Land Cover Classification Network

##### Source code of Rußwurm & Körner (2018) at [TBD]()

If you use this repository consider citing 
```
Rußwurm M., Körner M. (2018). Multi-Temporal Land Cover Classification with
Sequential Recurrent Encoders. ISPRS International Journal of Geo-Information, 2018.
```

## Dependencies

Implementations of ConvGRU and ConvLSTM forked from https://github.com/carlthome/tensorflow-convlstm-cell
```bash
git clone https://github.com/MarcCoru/tensorflow-convlstm-cell.git utils/convrnn
```

Python packages
```bash
conda install -y gdal
pip install psycopg2
pip install configobj
pip install matplotlib
pip install pandas
pip install configparser
```
## Jupyter notebooks


## Network training and evaluation

### On Local Machine (requires dependencies)

build network with 24 10m pixel graph
```bash
mkdir -p tmp

# build network graph (will be defined in <modelfolder>/graph.meta)
python modelzoo/seqencmodel.py --modelfolder tmp/convgru128 --convrnn_filters 128 --convcell gru --num_classes 17 --pix10m 24

# train network on 24px data of 2016 and 2017 (will write checkpoints <modelfolder>/model*)
python train.py tmp/convgru128 --datadir data_/datasets/240 -d 2016 --temporal_samples 30 --epochs 30 --shuffle True --batchsize 4 -d 2016 2017

# build 48px network with same specs
python modelzoo/seqencmodel.py --modelfolder tmp/convgru128_48px --convrnn_filters 128 --convcell gru --num_classes 17 --pix10m 48

# initialize new model graph (create empty checkpoint files)
python init_graph.py tmp/convgru128_48px/graph.meta

# optional: compare tensor dimensions of two graphs
python compare_graphs.py tmp/convgru128 tmp/convgru128_48px

# copy network weights from source (24px) network to target (48px) network
python copy_network_weights.py tmp/convgru128 tmp/convgru128_48px

# evaluate the networks on data of 2017
python evaluate.py tmp/convgru128 --datadir data_/datasets/240 --storedir tmp/eval/24 --writetiles --writeconfidences --batchsize 1 --dataset 2017
python evaluate.py tmp/convgru128_48px --datadir data_/datasets/480 --storedir tmp/eval/48 --writetiles --writeconfidences --batchsize 1 --dataset 2017

# optional: merge individual tiles to larger tifs
bash evaluate.sh tmp/eval/48
```

### From Docker Image (requires nvidia-docker)

```bash
# get path of current directory (https://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within)
thisdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# alias for command: start nvidia-docker session and forward folders for data and models
dockercmd="nvidia-docker run -ti -v $thisdir/data_/datasets:/data -v $thisdir/tmp:/model marccoru/mtlcc:latest"

## same operations as above. paths within docker container

# create model
$dockercmd python modelzoo/seqencmodel.py --modelfolder /model/convgru128 --convrnn_filters 128 --convcell gru --num_classes 17 --pix10m 24

# start training
$dockercmd python train.py /model/convgru128 --datadir /data/240 -d 2016 --temporal_samples 30 --epochs 30 --shuffle True --batchsize 4 -d 2016 2017

# evaluate
$dockercmd python evaluate.py /model/convgru128 --datadir /data/240 --storedir /model/24 --writetiles --writeconfidences --batchsize 1 --dataset 2017
```

## Extract Activations

extract internal activation images

```bash
python activations.py data_/models/convlstm256_48px/ data_/datasets/48 tmp/act -d 2016 -p eval -t 16494
```

## Docker support

### Build docker container
```bash
docker build -t mtlcc .
```

```bash
docker tag mtlcc:latest marccoru/mtlcc:latest
```

```bash
docker push marccoru/mtlcc:latest
```