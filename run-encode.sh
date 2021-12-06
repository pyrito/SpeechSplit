#!/bin/bash
LIBRISPEECH_PATH=/home/vkarthik/LibriSpeech_subset
SPEECHSPLIT_ASSETS=/home/vkarthik/SpeechSplit/assets

python3 make_spk.py

for mode in train dev test; do
    MODE=$mode
    rm -rf $SPEECHSPLIT_ASSETS/spmel
    rm -rf $SPEECHSPLIT_ASSETS/raptf0
    python3 make_spect_f0.py --mode $MODE
    python3 make_metadata.py --mode $MODE
    python3 main.py --encode-mode $MODE --resume_iters 660000 --model_save_dir /home/vkarthik
    mv $SPEECHSPLIT_ASSETS/encoded-$MODE.pkl $LIBRISPEECH_PATH/$MODE
done 