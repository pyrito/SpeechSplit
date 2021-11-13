import pickle
from os import path

ls_dir = path.join('assets', 'LibriSpeech')
speakers_file = path.join(ls_dir, 'SPEAKERS.TXT')
out_file = path.join('assets', 'ls_spk2gen.pkl')
speakers = {}

for line in open(speakers_file):
    if not line.startswith(';'):
        line_data = line.split('|')
        speakers[line_data[0].strip()] = line_data[1].strip()


pickle.dump(speakers, open(out_file, 'wb'))