# fcn-example

1. Download Pascal VOC 2012 and SDS data and unzip them in ~/data/
   Dowload `VGG_ILSVRC_16_layers.caffemodel` file and into this directory
2. To create database files and inital network run:
    python voc_db.py
    python fcn-voc-32s-init.py

3. For training run:
    ./train.sh
