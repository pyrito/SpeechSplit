import pickle
from os import path

ls_dir = path.join('/home/vkarthik', 'LibriSpeech_subset')
speakers_file = path.join(ls_dir, 'SPEAKERS.TXT')
out_file = path.join('assets', 'ls_spk2gen.pkl')
spk_emb_file = path.join('assets', 'ls_spk2emb.pkl')

spk2gen = {}
spk2emb = {}

for line in open(speakers_file):
    if not line.startswith(';'):
        line_data = line.split('|')
        id = line_data[0].strip()
        gender = line_data[1].strip()
        spk2gen[id] = gender
        spk2emb[id] = len(spk2gen)

pickle.dump(spk2gen, open(out_file, 'wb'))
pickle.dump(spk2emb, open(spk_emb_file, 'wb'))

