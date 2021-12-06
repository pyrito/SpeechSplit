#!/bin/bash

# Setup directories 
LIBRISPEECH_PATH=/home/vkarthik/LibriSpeech_subset

rm -rf $LIBRISPEECH_PATH/train/encoded
mkdir $LIBRISPEECH_PATH/train/encoded


rm -rf $LIBRISPEECH_PATH/dev/encoded
mkdir $LIBRISPEECH_PATH/dev/encoded

rm -rf $LIBRISPEECH_PATH/test/encoded
mkdir $LIBRISPEECH_PATH/test/encoded

# Run preprocessing
python3 util/librispeech_preprocess.py /home/vkarthik/LibriSpeech_subset/ train/ --dev_sets dev/ --tt_sets test/

# Run training
python3 train_libri.py config/las_libri_config.yaml
