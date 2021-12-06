import os
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()

# Preprocess configuration.
parser.add_argument('--mode', type=str, default="train", help='train, dev, or test')
args = parser.parse_args()
print(args)

rootDir = 'assets/spmel'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)


speakers = []
for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
    
    # use hardcoded onehot embeddings in order to be cosistent with the test speakers
    # modify as needed
    # may use generalized speaker embedding for zero-shot conversion
    # TODO(Rohan, Karthik): use ls_spk2emb to generate one hot embeddings for training (if needed)
    spkid = np.zeros((82,), dtype=np.float32)
    if speaker == 'p226':
        spkid[1] = 1.0
    else:
        spkid[7] = 1.0
    utterances.append(spkid)
    
    # create file list
    for fileName in sorted(fileList):
        utterances.append(os.path.join(speaker,fileName))
    speakers.append(utterances)
    
with open(os.path.join(rootDir, f'{args.mode}.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)    