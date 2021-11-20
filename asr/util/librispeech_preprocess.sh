if [ "$#" -ne 1 ]; then
    echo "Usage : ./librispeech_preprocess.sh <LibriSpeech folder>"
fi

python3 librispeech_preprocess.py $1 dev-clean/ --dev_sets dev-clean/ --tt_sets test-clean/
