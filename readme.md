# Pytorch -- Multitemporal Land Cover Classification Network

##### A (barebone) Pytorch port of Rußwurm & Körner (2018) Tensorflow implementation ((link)[https://github.com/TUM-LMF/MTLCC])

If you use this repository consider citing 
```
Rußwurm M., Körner M. (2018). Multi-Temporal Land Cover Classification with
Sequential Recurrent Encoders. ISPRS International Journal of Geo-Information, 2018.
```

## Dependencies

Python packages
```bash
conda install pytorch==0.4.1 torchvision==0.2.1 -c pytorch

pip install pandas>=0.23.4
pip install visdom==0.1.8.4
pip install rasterio>=1.0.2
```

<p align="center">
<img src="doc/lstm.gif" width="500" />
</p>

```
bash download.sh
```
