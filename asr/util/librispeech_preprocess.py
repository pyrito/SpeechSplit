
from pydub import AudioSegment
import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import scipy.io.wavfile as wav
from python_speech_features import logfbank
import pickle
import argparse


parser = argparse.ArgumentParser(description='Librispeech preprocess.')

parser.add_argument('root', metavar='root', type=str,
                     help='Absolute file path to LibriSpeech. (e.g. /usr/downloads/LibriSpeech/)')

parser.add_argument('tr_sets', metavar='tr_sets', type=str, nargs='+',
                     help='Training datasets to process in LibriSpeech. (e.g. train-clean-100/)')

# parser.add_argument('--encoded_path', metavar='encoded_path', type=str, default='',
#                     help='Path to encoded pickle files')

parser.add_argument('--dev_sets', metavar='dev_sets', type=str, nargs='+', default=[] ,
                     help='Validation datasets to process in LibriSpeech. (e.g. dev-clean/)')

parser.add_argument('--tt_sets', metavar='tt_sets', type=str, nargs='+', default=[] ,
                     help='Testing datasets to process in LibriSpeech. (e.g. test-clean/)')

parser.add_argument('--n_jobs', dest='n_jobs', action='store', default=-2 ,
                   help='number of cpu availible for preprocessing.\n -1: use all cpu, -2: use all cpu but one')
parser.add_argument('--n_filters', dest='n_filters', action='store', default=40 ,
                   help='number of filters for fbank. (Default : 40)')
parser.add_argument('--win_size', dest='win_size', action='store', default=0.025 ,
                   help='window size during feature extraction (Default : 0.025 [25ms])')
parser.add_argument('--norm_x', dest='norm_x', action='store', default=False ,
                   help='Normalize features s.t. mean = 0 std = 1')


paras = parser.parse_args()

root = paras.root
train_path = paras.tr_sets
dev_path = paras.dev_sets
test_path = paras.tt_sets
n_jobs = paras.n_jobs
n_filters = paras.n_filters
win_size = paras.win_size
norm_x = paras.norm_x
# encoded_path = paras.encoded_path

def traverse(root,path,search_fix='.flac',return_label=False):
    f_list = []

    for p in path:
        p = root + p
        for sub_p in sorted(os.listdir(p)):
            for sub2_p in sorted(os.listdir(p+sub_p+'/')):
                if return_label:
                    # Read trans txt
                    with open(p+sub_p+'/'+sub2_p+'/'+sub_p+'-'+sub2_p+'.trans.txt','r') as txt_file:
                        for line in txt_file:
                            f_list.append(' '.join(line[:-1].split(' ')[1:]))
                else:
                    # Read acoustic feature
                    for file in sorted(os.listdir(p+sub_p+'/'+sub2_p)):
                        if search_fix in file:
                            file_path = p+sub_p+'/'+sub2_p+'/'+file
                            f_list.append(file_path)   
    return f_list

def get_labels_for_files(ground_truth_files, librispeech_root):
    labels = []
    for gt_file in ground_truth_files:
        step_1 = gt_file.split('-')[0]
        step_2 = gt_file.split('-')[1]
        with open(os.path.join(librispeech_root, step_1, gt_file)+'.trans.txt') as f:
            for line in f:
                labels.append(' '.join(line[:-1].split(' ')[1:]))
    return labels

def save_and_get_encoded_tensors(path, match, save_root):
    f_list = []
    ground_truth_files = set([])
    for sub_p in sorted(os.listdir(path)):
        if sub_p == match:
            with open(os.path.join(path, sub_p), 'rb') as f:
                encoded = pickle.load(f)
            for k,v in sorted(encoded.items()):
                file_name = k + '.npy'
                ground_truth_file = '-'.join(file_name.split('-')[:-1])
                print(k)
                print(ground_truth_file)
                full_dst_path = os.path.join(save_root, file_name)
                np.save(full_dst_path, np.squeeze(v.detach().numpy(), axis=0))
                f_list.append(full_dst_path)
                ground_truth_files.add(ground_truth_file)

    return f_list, sorted(list(ground_truth_files))
    

print('----------Processing Datasets----------')
print('Training sets :',train_path)
print('Validation sets :',dev_path)
print('Testing sets :',test_path)

# # log-mel fbank 2 feature
print('---------------------------------------') 
print('Preparing Training Dataset...',flush=True)


# Read the pickle file, save the individual tensors into some file, return a list of these files.
# Using the file name that I see in the pickle file, figure out the labels, use the same code
# Write everything out 
# print(train_path)
train_dir = os.path.join(root, train_path[0])
train_encoded_save_path = os.path.join(train_dir, 'encoded')
tr_file_list, ground_truth_files = save_and_get_encoded_tensors(train_dir, "encoded-train.pkl", train_encoded_save_path)          
tr_text = get_labels_for_files(ground_truth_files, train_dir)

assert len(tr_file_list) == len(tr_text)

# Create char mapping
char_map = {}
char_map['<sos>'] = 0
char_map['<eos>'] = 1
char_idx = 2

# map char to index
for text in tr_text:
    for char in text:
        if char not in char_map:
            char_map[char] = char_idx
            char_idx += 1
            
# Reverse mapping
rev_char_map = {v:k for k,v in char_map.items()}

# Save mapping
with open(root+'idx2chap.csv','w') as f:
    f.write('idx,char\n')
    for i in range(len(rev_char_map)):
        f.write(str(i)+','+rev_char_map[i]+'\n')

# text to index sequence
tmp_list = []
for text in tr_text:
    tmp = []
    for char in text:
        tmp.append(char_map[char])
    tmp_list.append(tmp)
tr_text = tmp_list
del tmp_list

# write dataset
file_name = 'train.csv'

print('Writing dataset to '+root+file_name+'...',flush=True)

with open(root+file_name,'w') as f:
    f.write('idx,input,label\n')
    for i in range(len(tr_file_list)):
        f.write(str(i)+',')
        f.write(tr_file_list[i]+',')
        for char in tr_text[i]:
            f.write(' '+str(char))
        f.write('\n')

print()
print('Preparing Validation Dataset...',flush=True)

dev_dir = os.path.join(root, dev_path[0])
dev_encoded_save_path = os.path.join(dev_dir, 'encoded')
dev_file_list, ground_truth_files = save_and_get_encoded_tensors(dev_dir, "encoded-dev.pkl", dev_encoded_save_path)          
dev_text = get_labels_for_files(ground_truth_files, dev_dir)

assert len(dev_file_list) == len(dev_text)

# text to index sequence
tmp_list = []
for text in dev_text:
    tmp = []
    for char in text:
        tmp.append(char_map[char])
    tmp_list.append(tmp)
dev_text = tmp_list
del tmp_list

# write dataset
file_name = 'dev.csv'

print('Writing dataset to '+root+file_name+'...',flush=True)

with open(root+file_name,'w') as f:
    f.write('idx,input,label\n')
    for i in range(len(dev_file_list)):
        f.write(str(i)+',')
        f.write(dev_file_list[i]+',')
        for char in dev_text[i]:
            f.write(' '+str(char))
        f.write('\n')

print()
print('Preparing Testing Dataset...',flush=True)

tt_dir = os.path.join(root, test_path[0])
tt_encoded_save_path = os.path.join(tt_dir, 'encoded')
tt_file_list, ground_truth_files = save_and_get_encoded_tensors(tt_dir, "encoded-test.pkl", tt_encoded_save_path)          
tt_text = get_labels_for_files(ground_truth_files, tt_dir)

assert len(tt_file_list) == len(tt_text)

# text to index sequence
tmp_list = []
for text in tt_text:
    tmp = []
    for char in text:
        tmp.append(char_map[char])
    tmp_list.append(tmp)
tt_text = tmp_list
del tmp_list

# write dataset
file_name = 'test.csv'

print('Writing dataset to '+root+file_name+'...',flush=True)

with open(root+file_name,'w') as f:
    f.write('idx,input,label\n')
    for i in range(len(tt_file_list)):
        f.write(str(i)+',')
        f.write(tt_file_list[i]+',')
        for char in tt_text[i]:
            f.write(' '+str(char))
        f.write('\n')

