import numpy as np
import os
import sys 
sys.path.append('./utils')
from glob import glob
import pretty_midi 
import h5py
import shutil
import copy
import subprocess
import argparse

from utils.parse_utils import *


'''
Save Batches for Train/Val/Test 

CMD: https://github.com/shiehn/chord-melody-dataset
HLSD: https://github.com/wayne391/lead-sheet-dataset (download "Source" file)

** Save the original raw datasets at the directory 
   where this code belongs to:
    --> /(codeDiretory)/CMD/dataset 
     --> /(codeDiretory)/HLSD/dataset 

** COMMANDS **
> Save CMD batches from raw data
- python3 process_data.py --dataset CMD 

> Save HLSD batches from raw data
- python3 process_data.py --dataset HLSD
'''

class FeatureIndex(object):

    def __init__(self, dataset):

        # dict for features-to-indices
        if dataset == "CMD":
            uniq_chords = np.load("unique_chord_labels_CMD.npy").tolist()[:-1]
            self.uniq_chords_simple = self.simplify_all_chord_labels_CMD(uniq_chords)
        elif dataset == "HLSD":
            uniq_chords = np.load("unique_chord_labels_HLSD.npy").tolist()
            self.uniq_chords_simple = self.simplify_all_chord_labels_HLSD(uniq_chords)            

        self.type2ind = self.feature2ind(['16th', 'eighth', 'eighth_dot', 
            'quarter', 'quarter_dot', 'half', 'half_dot', 'whole', 'none'])
        self.tied2ind = self.feature2ind(['start', 'stop', 'none'])
        self.chord2ind_func = self.feature2ind(uniq_chords)
        self.chord2ind_func_simple = self.feature2ind(self.uniq_chords_simple)
        self.root2ind = self.feature2ind(['C','C#','D',
            'D#','E','F','F#','G','G#','A','A#','B'])
        self.ind2chord_func = self.ind2feature(uniq_chords)
        self.ind2chord_func_simple = self.ind2feature(self.uniq_chords_simple)
        self.ind2root = self.ind2feature(['C','C#','D',
            'D#','E','F','F#','G','G#','A','A#','B'])

    def feature2ind(self, features):
        f2i = dict()
        for i, f in enumerate(features):
            f2i[f] = i 
        return f2i

    def ind2feature(self, features):
        i2f = dict()
        for i, f in enumerate(features):
            i2f[i] = f 
        return i2f

    def simplify_all_chord_labels_CMD(self, uniq_chords):
        uniq_chords_simple = list()
        for c in uniq_chords:
            new_lab = self.simplify_chord_label_CMD(c)
            uniq_chords_simple.append(new_lab)
        return np.unique(uniq_chords_simple)

    def simplify_all_chord_labels_HLSD(self, uniq_chords):
        uniq_chords_simple = list()
        for c in uniq_chords:
            new_lab = self.simplify_chord_label_HLSD(c)
            uniq_chords_simple.append(new_lab)
        return np.unique(uniq_chords_simple)

    def simplify_chord_label_CMD(self, c):
        labs = c.split("_")
        lab = labs[0]
        if lab != "":
            if "9" in lab:
                if "9" == lab[0]: # dominant
                    lab = "7"
                else:
                    lab = lab.replace("9", "7")
            if "dim7" == lab:
                lab = "dim"
        new_c = "{}_".format(lab)
        return new_c

    def simplify_chord_label_HLSD(self, c):
        labs = c.split("_")
        lab = labs[0]
        if lab != "":
            if "9" == lab or "11" == lab: # dominant
                lab = "7"
            else:
                if "9" in lab:
                    lab = lab.replace("9", "7")
                elif "11" in lab:
                    lab = lab.replace("11", "7")
            if "ø" in lab or "o" in lab:
                lab = "dim5"
        new_c = "{}_".format(lab)
        return new_c


sep = os.sep

def ind2str(ind, n):
    ind_ = str(ind)
    rest = n - len(ind_)
    str_ind = rest*"0" + ind_
    return str_ind 

def split_sets_CMD():
    '''
    * Total 473 songs
        - 8:1:1
        - 84 songs with no full transposed versions --> val/test 
        - 389 songs (x 12 transposed) --> train
    '''
    datapath = sep.join(['.','CMD','output'])
    all_songs = sorted(glob(os.path.join(datapath, "*"+sep)))

    test_list = list()
    for c in all_songs:
        pieces = sorted(glob(os.path.join(c, 'features.*.npy')))
        if len(pieces) < 12:
            test_list.append(c)
    test_list = sorted(test_list)
    val_songs = test_list[::2]
    test_songs = test_list[1::2]

    train_path = sep.join([".","CMD","exp","train","raw"])
    val_path = sep.join([".","CMD","exp","val","raw"])
    test_path = sep.join([".","CMD","exp","test","raw"])

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    for c in all_songs:
        pieces = sorted(glob(os.path.join(c, 'features.*.npy')))
        c_name = c.split(sep)[-2]
        if c in val_songs:
            savepath = val_path
        elif c in test_songs:
            savepath = test_path
        else: savepath = train_path
        savepath_ = os.path.join(savepath, c_name)
        if not os.path.exists(savepath_):
            os.makedirs(savepath_)
        for p in pieces:
            p_name = os.path.basename(p).split('.')[-2] # transposed key
            shutil.copy(p,
                os.path.join(savepath_, "features.{}.{}.npy".format(c_name, p_name)))
            print("saved xml data for {}/{}".format(c_name, p_name))

def split_sets_HLSD():
    '''
    * Total 13335 parts
        - 9218 songs
        - only 4/4
    '''
    datapath = sep.join(['.','HLSD','output','event'])
    all_songs = sorted(glob(os.path.join(datapath, "*"+sep)))

    file_list = list()
    song_num = 0
    for c in all_songs:
        pieces = sorted(glob(os.path.join(c, '*'+sep)))
        for p in pieces:
            songs = sorted(glob(os.path.join(p, '*'+sep)))
            for s in songs:
                song_num += 1
                parts = sorted(glob(os.path.join(s, 'features.*.npy')))
                for part in parts:
                    file_list.append(part)

    file_num = len(file_list)
    train_num = int(file_num*0.8) # originally: file_num - 1000
    val_num = (file_num - train_num) // 2
    train_path = sep.join(['.','HLSD','exp','train','raw'])
    val_path = sep.join(['.','HLSD','exp','val','raw'])
    test_path = sep.join(['.','HLSD','exp','test','raw'])

    val_songs = file_list[train_num:train_num+val_num]
    test_songs = file_list[train_num+val_num:]

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    for c in file_list:
        if c in val_songs:
            savepath = val_path
        elif c in test_songs:
            savepath = test_path
        else: savepath = train_path
        c_name = '_'.join(os.path.basename(c).split('.')[1:-1])
        shutil.copy(c,
            os.path.join(savepath, "features.{}.npy".format(c_name)))
        print("saved xml data for {}".format(c_name))

def make_align_matrix(roll, attacks_ind):
    new_ind = attacks_ind - np.min(attacks_ind) # start from 0
    align_mat = np.zeros([roll.shape[0], len(attacks_ind)])
    new_ind = np.concatenate([new_ind, [len(roll)]], axis=0)
    onset = 0
    for i in range(len(new_ind)-1):
        start = new_ind[i]
        end = new_ind[i+1]
        align_mat[start:end, onset] = 1
        # print(start, end, onset)
        onset += 1
    return align_mat

def make_align_matrix_roll(data, durs, attacks_ind):
    roll = np.zeros([np.sum(durs), 88])
    new_ind = attacks_ind - np.min(attacks_ind) # start from 0
    align_mat = np.zeros([roll.shape[0], len(attacks_ind)])   
    note_mat = np.zeros([roll.shape[0], 1])   
    new_ind = np.concatenate([new_ind, [len(roll)]], axis=0)
    '''
    0-13: pitch-class 
    13-21: beat 
    21-33: key 
    33-41: octave 
    41-49: type
    49-51: cnew
    '''
    start = 0
    onset = 0
    for i in range(len(new_ind)-1):
        start_note = new_ind[i]
        end_note = new_ind[i+1]        
        note = data[start_note:end_note]
        dur = durs[start_note:end_note]

        align_mat[start:start+np.sum(dur), onset] = 1
        onset += 1 

        for n, d in zip(note, dur):
            pc = np.argmax(n[:13], axis=-1)
            octave = np.argmax(n[33:41], axis=-1)
            pitch = np.where(pc==12, 108, pc + 12 * (octave + 1))

            roll[start:start+d, pitch-21] = 1
            note_mat[start,0] = 1
            start += d
    
    return align_mat, roll, note_mat

def make_align_matrix_note(data, durs):
    roll = np.zeros([np.sum(durs), 89])
    note_mat = np.zeros([roll.shape[0], len(data)])
    '''
    * data index list *
    0-13: pitch-class 
    13-21: beat 
    21-33: key 
    33-42: octave 
    42-50: type
    50-52: cnew

    * pitch 
    A0 ~ C8 = 21 ~ 108 
    pitch = (pc % 12) + 12 * (octave + 1)
    '''
    start = 0
    onset = 0
    for n, d in zip(data, durs):
        pc = np.argmax(n[:13], axis=-1)
        octave = np.argmax(n[33:42], axis=-1)
        pitch = (pc % 12) + 12 * (octave + 1)

        roll[start:start+d, pitch-21] = 1
        note_mat[start:start+d, onset] = 1
        start += d
        onset += 1
    
    return note_mat, roll

def make_align_matrix_roll2note(data, nnew):
    note_ind = [i for i, n in enumerate(nnew) if n == 1]
    note_mat = np.zeros([len(data), len(note_ind)])
    start = 0
    onset = 0
    for i in range(len(note_ind)):
        start = note_ind[i] 
        if i < len(note_ind)-1:
            end = note_ind[i+1]
        elif i == len(note_ind)-1:
            end = len(data) 
        note_mat[start:end, onset] = 1
        onset += 1
    
    return note_mat

def make_align_matrix_note2chord(nnew, cnew):
    note_ind = [i for i, n in enumerate(nnew) if n == 1]
    chord_ind = [i for i, n in enumerate(cnew) if n == 1]
    chord_mat = np.zeros([len(note_ind), len(chord_ind)])
    start = -1
    onset = -1
    for i in range(len(nnew)):
        if nnew[i] == 1:
            start += 1
            if cnew[i] == 1:
                onset += 1
                chord_mat[start, onset] = 1
            elif cnew[i] == 0:
                chord_mat[start, onset] = 1 
        elif nnew[i] == 0:
            if cnew[i] == 1:
                onset += 1
    
    return chord_mat
        
def check_chord_root(chord_root):
    new_root = None
    if chord_root == 'Ab':
        new_root = 'G#'
    elif chord_root == 'Bb':
        new_root = 'A#'
    elif chord_root == 'B#':
        new_root = 'C'
    elif chord_root == 'Cb':
        new_root = 'B'
    elif chord_root == 'Db':
        new_root = 'C#'
    elif chord_root == 'E#':
        new_root = 'F'
    elif chord_root == 'Eb':
        new_root = 'D#'
    elif chord_root == 'Fb':
        new_root = 'E'
    elif chord_root == 'Gb':
        new_root = 'F#' 
    else:
        new_root = chord_root

    return new_root

def get_chord_notes(chord_kind, chord_root, pc_norm=True):
    '''
    chord root should be in int
    '''
    kind1, kind2 = chord_kind.split("_")
    if kind1 == '7': # dominant 7
        chord_notes = [chord_root, chord_root+4, 
            chord_root+7, chord_root+10]
    elif kind1 == '9': # dominant 9
        chord_notes = [chord_root, chord_root+4, 
            chord_root+7, chord_root+10, chord_root+14]
    elif kind1 == '' or kind1 == '5': # major 3
        chord_notes = [chord_root, chord_root+4, 
            chord_root+7]
    elif kind1 == 'dim7': # diminished 7
        chord_notes = [chord_root, chord_root+3, 
            chord_root+6, chord_root+9]
    elif kind1 == 'dim' or kind1 == 'dim5': # diminished 3
        chord_notes = [chord_root, chord_root+3, 
            chord_root+6]        
    elif kind1 == 'm7': # minor 7
        chord_notes = [chord_root, chord_root+3, 
            chord_root+7, chord_root+10]        
    elif kind1 == 'm' or kind1 == 'm5': # minor 3
        chord_notes = [chord_root, chord_root+3, 
            chord_root+7]
    elif kind1 == 'maj7': # major 7
        chord_notes = [chord_root, chord_root+4, 
            chord_root+7, chord_root+11]
    elif kind1 == 'maj9': # major 9
        chord_notes = [chord_root, chord_root+4, 
            chord_root+7, chord_root+11, chord_root+14]
    
    if kind2 == 'b9': # flat 9 (only when D7)
        chord_notes += [chord_root+13] 
    elif kind2 == 'b5': # flat 5
        chord_notes[2] -= 1 

    if pc_norm is True:
        # norm into [chord root ~ chord root+11]
        chord_notes_norm = list()
        for c in chord_notes: 
            normed_tone = np.mod(c, 12)
            chord_notes_norm.append(normed_tone)
        chord_notes_final = chord_notes_norm 
    else:
        chord_notes_final = chord_notes 

    return chord_notes_final

def get_chord_notes_simple(chord_kind, chord_root, pc_norm=True):
    '''
    chord root should be in int
    '''
    kind1, kind2 = chord_kind.split("_")
    if kind1 == '7': # dominant 7
        chord_notes = [chord_root, chord_root+4, 
            chord_root+7, chord_root+10]
    elif kind1 == '' or kind1 == '5': # major 3
        chord_notes = [chord_root, chord_root+4, 
            chord_root+7]
    elif kind1 == 'dim' or kind1 == 'dim5': # diminished 3
        chord_notes = [chord_root, chord_root+3, 
            chord_root+6]        
    elif kind1 == 'm7': # minor 7
        chord_notes = [chord_root, chord_root+3, 
            chord_root+7, chord_root+10]        
    elif kind1 == 'm' or kind1 == 'm5': # minor 3
        chord_notes = [chord_root, chord_root+3, 
            chord_root+7]
    elif kind1 == 'maj7': # major 7
        chord_notes = [chord_root, chord_root+4, 
            chord_root+7, chord_root+11]

    if pc_norm is True:
        # norm into [chord root ~ chord root+11]
        chord_notes_norm = list()
        for c in chord_notes: 
            normed_tone = np.mod(c, 12)
            chord_notes_norm.append(normed_tone)
        chord_notes_final = chord_notes_norm 
    else:
        chord_notes_final = chord_notes 

    return chord_notes_final

def decide_chord_tone(chord_kind, chord_root, pitch, pc_norm=True):
    # get set of chord tones
    chord_tones = get_chord_notes(
        chord_kind, chord_root, pc_norm=pc_norm)

    # decide if chord tone
    if pitch in chord_tones:
        is_chord_tone = True 
    else:
        is_chord_tone = False

    return is_chord_tone, chord_tones

def decide_ct_simple(chord_kind, chord_root, pitch, pc_norm=False):
    # get set of chord tones
    chord_tones = get_chord_notes_simple(
        chord_kind, chord_root, pc_norm=pc_norm)

    # decide if chord tone
    if pitch in chord_tones:
        is_chord_tone = True 
    else:
        is_chord_tone = False

    return is_chord_tone, chord_tones

def make_pianorolls_with_onset(notes, measures, measures_dict, inds, chord_type=None):

    FI = FeatureIndex(dataset="CMD")

    maxlen_time = measures[-1][0] + 1 # last measure's last beat + beat_unit 
    unit = 0.5 / 4 # 16th note 
    maxlen = int(maxlen_time // unit)
    note_roll = np.zeros([maxlen, 89])
    key_roll = np.zeros([maxlen, 12])
    if chord_type == "all":
        chord_roll = np.zeros([maxlen, 156])
    elif chord_type == "simple":
        chord_roll = np.zeros([maxlen, 72])
    onset_roll = np.zeros([maxlen, 2]) # onsets for note & chord 
    onset_roll_xml = np.zeros([maxlen, 2]) 
    beat_roll = np.zeros([maxlen, 1]) # 3-stong / 2-mid / 1-weak / 0-none
    note_ind_onset = list()

    # note roll 
    for i in range(len(notes)):
        # get onset and offset
        note_ind = inds[i]
        onset = notes[i][0]
        if i < len(notes)-1:
            offset = notes[i+1][0]
        elif i == len(notes)-1:
            offset = maxlen_time 
        note_ind_onset.append([note_ind, int(onset // unit)])
        onset_roll_xml[int(onset // unit), 0] = 1

        # if note lasts over corresponding measure(1/2)
        note_chunks = list()
        for m in range(len(measures)):
            measure_onset = measures[m][0]
            if m < len(measures)-1:
                next_measure_onset = measures[m+1][0]
                if onset >= measure_onset and onset < next_measure_onset:
                    # print(onset, measure_onset, next_measure_onset)
                    new_onset = copy.deepcopy(onset)
                    next_measures = measures[m+1:] + [[maxlen_time]] 
                    for n in next_measures:
                        if offset <= n[0]: # ends within a measure
                            note_chunks.append([new_onset, offset])
                            break 
                        elif offset > n[0]: # longer than a measure
                            note_chunks.append([new_onset, n[0]])
                            new_onset = n[0] # update onset
                    break
                else:
                    continue

            elif m == len(measures)-1: # last measure
                next_measure_onset = maxlen_time
                note_chunks.append([onset, offset])
            
        # print(onset, offset, note_chunks)
                    
        for each in note_chunks:
            start = int(each[0] // unit)
            end = int(each[1] // unit)
            # get pitch 
            pitch = notes[i][1] 
            if pitch is None:
                pitch = 88 
            else:
                pitch = pitch - 21 
            # get key
            key = notes[i][2]
                
            # put values to rolls 
            note_roll[start:end, pitch] = 1
            onset_roll[start, 0] = 1
            key_roll[start:end, key] = 1 

    '''
    4/4 measure -->> 16 units 
    0: strong / 4: weak / 8: mid / 12: weak   
    '''
    for measure in measures_dict:
        beats = measure[1]['beat']
        for b in beats:
            ind = b['index']
            onset = int(b['time'] // unit)
            # if ind == 0: # strong 
            #     beat_roll[onset, 0] = 3 
            # elif ind == 2: # middle 
            #     beat_roll[onset, 0] = 2 
            # elif ind == 1 or ind == 3: # weak 
            #     beat_roll[onset, 0] = 1 
            if ind == 0 or ind == 2: # strong 
                beat_roll[onset, 0] = 2 
            else: 
                beat_roll[onset, 0] = 1 

    # chord roll
    for i in range(len(measures)):
        if measures[i][3] == '':
            continue 
        else:
            # get onset and offset
            start = int(measures[i][0] // unit) 
            if i < len(measures)-1:
                end = int(measures[i+1][0] // unit)
            elif i == len(measures)-1:
                end = maxlen   
            # get chord      
            if chord_type == "all":
                chord_kind = "{}_{}".format(measures[i][3]['kind'], measures[i][3]['degrees'])
                chord2ind = FI.chord2ind_func # function
                chord_len = len(uniq_chords)
            elif chord_type == "simple":
                chord_kind = FI.simplify_chord_label_CMD(measures[i][3]['kind'])
                chord2ind = FI.chord2ind_func_simple # function
                chord_len = len(FI.uniq_chords_simple)
            chord_root = check_chord_root(measures[i][3]['root'])

            ckind = chord2ind[chord_kind] # chord root
            croot = FI.root2ind[chord_root] # chord root
            chord = croot * chord_len + ckind # 156 classes / 36 classes
            chord_roll[start:end, chord] = 1 
            onset_roll[start, 1] = 1 
            onset_roll_xml[start, 1] = 1
    
    return note_roll, chord_roll, key_roll, beat_roll, onset_roll, onset_roll_xml, note_ind_onset

def get_roll_CMD(features, chord_type=None):
    '''
    This function is especially for Chord-Melody-Dataset
    in which each measure includes 2 chords in 1/2 measure length
    (all measures in 4/4, 120 BPM)
    '''
    notes, measures = features
    # gather note info
    note_list = list()
    ind_list = list()
    for note in notes:
        note_list.append(
            [note[1]['time_position'], 
             note[1]['pitch_abs']['num'], 
             note[1]['key']['root']])
        ind_list.append(note[0])
    # gather measure info
    measure_list = list()
    prev_chord = None
    for measure in measures:
        chords = measure[1]['chord']
        beats = measure[1]['beat']
        measure_num = measure[1]['measure_num']
        for beat in beats:
            beat['chord'] = ''

            if beat['index'] == 0: 
                if len(chords) == 0:
                    if prev_chord is not None: # no candidate for first chord
                        beat['chord'] = prev_chord
                elif len(chords) > 0:
                    '''
                    Some songs (ex. Nuage) include measures with 
                    2 chords that appear simultaneously 
                    probably because there is only one note (ex. whole note)
                    In this case, pick whatever that is parsed first 
                    as the first chord.
                    '''
                    if chords[0]['kind'] == 'none':
                        if prev_chord is not None: # no candidate for first chord
                            beat['chord'] = prev_chord
                    else:
                        beat['chord'] = chords[0]
                        prev_chord = chords[0]
                measure_list.append([
                    beat['time'], beat['index'], measure_num, beat['chord']])

            elif beat['index'] == 2:
                if len(chords) == 0 or len(chords) == 1: # no candidate for second chord
                    if prev_chord is not None:
                        beat['chord'] = prev_chord
                elif len(chords) > 1:
                    if chords[1]['kind'] == 'none':
                        if prev_chord is not None: # no candidate for first chord
                            beat['chord'] = prev_chord
                    else:
                        beat['chord'] = chords[1]
                        prev_chord = chords[1]
                measure_list.append([
                    beat['time'], beat['index'], measure_num, beat['chord']])

            else:
                continue
        
    note_roll, chord_roll, key_roll, beat_roll, onset_roll, onset_roll_xml, note_ind_onset = \
        make_pianorolls_with_onset(note_list, measure_list, measures, ind_list, chord_type=chord_type)

    return note_roll, chord_roll, key_roll, beat_roll, onset_roll, onset_roll_xml, note_ind_onset

def get_roll_HLSD(features, chord_type=None):

    FI = FeatureIndex(dataset="HLSD")

    orig_key = features['orig_key_info']
    key_root = FI.root2ind[check_chord_root(orig_key[0])]
    if orig_key[1] == '1':
        mode = 0 # major
    elif orig_key[1] == '6':
        mode = 1
    elif orig_key[1] in ['4', '5']: # lydian, mixolydian
        mode = 0 # major 
    else: # other modes
        mode = 1 
    key_sig = 12 * mode + key_root
    time_sig = features['time_signature']
    assert time_sig == '4'

    notes = features["melody"]
    measures = features["chord"]

    maxlen_time = measures[-1]['end'] 
    unit = 1 / 4 # 16th note 
    maxlen = int(maxlen_time // unit)
    note_roll = np.zeros([maxlen, 89])
    if chord_type == "all":
        chord_roll = np.zeros([maxlen, 156])
    elif chord_type == "simple":
        chord_roll = np.zeros([maxlen, 72])
    onset_roll = np.zeros([maxlen, 2]) # onsets for note & chord 
    onset_roll_xml = np.zeros([maxlen, 2]) # onsets before notes are chunked
    # beat_roll = np.zeros([maxlen, 1]) # 3-stong / 2-mid / 1-weak / 0-none
    note_ind_onset = list()

    # note roll
    prev_end = 0
    for i in range(len(notes)):
        # get onset and offset
        onset = notes[i]['start']
        offset = notes[i]['end']
        if onset > offset:
            continue
        if onset < prev_end:
            continue
        note_ind_onset.append([i, int(onset // unit)])
        onset_roll_xml[int(onset // unit), 0] = 1

        # if note lasts over corresponding measure(1/2)
        note_chunks = list()
        for m in range(len(measures)):
            measure_onset = measures[m]['start']
            if m < len(measures)-1:
                next_measure_onset = measures[m]['end']
                if onset >= measure_onset and onset < next_measure_onset:
                    # print(onset, measure_onset, next_measure_onset)
                    new_onset = copy.deepcopy(onset)
                    next_measures = measures[m+1:] 
                    for n in next_measures:
                        if offset <= n['start']: # ends within a measure
                            note_chunks.append([new_onset, offset])
                            break 
                        elif offset > n['start']: # longer than a measure
                            note_chunks.append([new_onset, n['start']])
                            new_onset = n['start'] # update onset
                    break
                else:
                    continue

            elif m == len(measures)-1: # last measure
                next_measure_onset = maxlen_time
                note_chunks.append([onset, offset])
                    
        for each in note_chunks:
            start = int(quantize(each[0], unit=0.25) // unit)
            end = int(quantize(each[1], unit=0.25) // unit)
            # get pitch 
            pitch = notes[i]['pitch'] 
            is_rest = notes[i]['pitch'] 
            if pitch == 0 or notes[i]['is_rest'] is True:
                pitch = 88 
            else:
                pitch = (pitch - 21) - key_root # normalize to C Major
                
            # put values to rolls 
            note_roll[start:end, int(pitch)] = 1
            onset_roll[start, 0] = 1
        
        prev_end = offset
        # print(onset, offset, note_chunks)

    # chord roll
    prev_end = 0
    for i in range(len(measures)):
        # get onset and offset
        start = int(measures[i]['start'] // unit) 
        end = int(measures[i]['end'] // unit)
        if i < len(measures)-1:
            next_start = int(measures[i+1]['start'] // unit) 
        elif i == len(measures)-1:
            next_start = int(measures[i]['end'] // unit)
        if end < next_start:
            end = next_start

        # get chord      
        chord_kind = "{}_".format(measures[i]['quality']+measures[i]['type'])
        if chord_type == "all":
            chord2ind = FI.chord2ind_func # function
            chord_len = len(uniq_chords)
        elif chord_type == "simple":
            chord_kind = FI.simplify_chord_label_HLSD(chord_kind)
            chord2ind = FI.chord2ind_func_simple # function
            chord_len = len(FI.uniq_chords_simple)
        
        croot = measures[i]['root'] - key_root # normalize to C Major
        if croot < 0:
            croot = (croot + 12) % 12
        ckind = chord2ind[chord_kind] # chord root
        chord = croot * chord_len + ckind # 156 classes / 36 classes
        chord_roll[start:end, chord] = 1  
        onset_roll[start, 0] = 1
        onset_roll[start, 1] = 1 
        onset_roll_xml[start, 1] = 1
    
    # cut if melody ends early (half-measure)
    onset_ind = np.where(onset_roll[:,1] == 1)[0].tolist() + [len(onset_roll)]
    for o in reversed(onset_ind):
        if o > 0:
            if np.sum(note_roll[o-8:o]) == 0:
                continue 
            elif np.sum(note_roll[o-8:o]) > 0:
                break 
    note_roll = note_roll[:o]
    onset_roll = onset_roll[:o]

    # fill in "rest" 
    j = 1
    non_zero = 0
    for i in range(len(note_roll)):
        frame = note_roll[i]
        if np.sum(frame) == 0:
            j += 1 
            continue 
        elif np.sum(frame) > 0:
            if np.sum(note_roll[non_zero+1]) == 0:
                note_roll[non_zero+1:non_zero+j, 88] = 1
                onset_roll[non_zero+1, 0] = 1
            j = 1
            non_zero = i
    if non_zero < len(note_roll)-1:
        if np.sum(note_roll[non_zero+1]) == 0:
            note_roll[non_zero+1:non_zero+j, 88] = 1
            onset_roll[non_zero+1, 0] = 1

    return note_roll, chord_roll, mode, onset_roll, onset_roll_xml, note_ind_onset

def save_batches_CMD(chord_type='simple'):    
    print("Saving batches...")

    parent_path = sep.join(['.','CMD','exp'])
    orig_path = sep.join(['.','CMD','dataset'])
    groups = sorted(glob(os.path.join(parent_path, "*"+sep)))
    maxlen, hop = 16, 8
    chord_list = list()

    for g, group in enumerate(groups): # train/val/test
        # group = groups[1]
        datapath = os.path.join(group, 'raw')
        savepath = os.path.join(group, "batch")
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        categs = sorted(glob(os.path.join(datapath, '*'+sep)))
        if len(categs) == 0:
            print("no feature files in {}".format(os.path.dirname(datapath)))
            continue

        for c, categ in enumerate(categs):
            # categ = categs[0]
            c_name = categ.split(sep)[-2]
            pieces = sorted(glob(os.path.join(categ, 'features.*.npy')))

            for piece in pieces:
                # piece = pieces[0]
                p_name = os.path.basename(piece).split('.')[-2]
                features = np.load(piece, allow_pickle=True).tolist()

                # collect chord types
                notes, measures = features
                for measure in measures:
                    chords = measure[1]['chord']
                    for chord in chords:
                        if chord['kind'] != 'none':
                            chord_name = "{}_{}".format(chord['kind'], chord['degrees'])
                            if chord_name not in chord_list:
                                chord_list.append(chord_name)
                # print(p_name)                

                # get onehot data
                inp, oup, key, beat, onset, onset_xml, inds = get_roll_CMD(features, chord_type=chord_type)
                # get indices where new chord
                new_chord_ind = [i for i, c in enumerate(onset[:,1]) if c == 1]
                new_chord_ind.append(len(inp))

                # make sure new_chord_inds are equally distanced
                prev_ind = None 
                for c in new_chord_ind: 
                    if prev_ind is not None:
                        # print(c - prev_ind) 
                        assert c - prev_ind == 8
                    prev_ind = c

                # save batch 
                num = 1
                for b in range(0, len(new_chord_ind), hop):

                    ind = ind2str(num, 3)
                    chord_ind = new_chord_ind[b:b+maxlen+1]
                    start, end = chord_ind[0], chord_ind[-1] # (maxlen+1)th chord
                    in1_ = inp[start:end]
                    key_ = key[start:end]
                    beat_ = beat[start:end]
                    nnew_ = onset[start:end, :1]
                    cnew_ = onset[start:end, -1:]
                    nnew2_ = onset_xml[start:end, :1]
                    cnew2_ = onset_xml[start:end, -1:]
                    
                    note_ind = [i for i, n in enumerate(nnew_) if n == 1]
                    # roll2note 
                    in2_ = make_align_matrix_roll2note(in1_, nnew_)
                    # note2chord
                    in3_ = make_align_matrix_note2chord(nnew_, cnew_)
                    
                    # get data in different units 
                    key_note = np.asarray([key_[n] for n in note_ind])
                    beat_note = np.asarray([beat_[n] for n in note_ind])
                    cnew_note = np.asarray([cnew_[n] for n in note_ind])
                    in4_ = np.concatenate([key_note, cnew_note, beat_note], axis=-1)
                    out1_ = np.asarray([oup[c] for c in chord_ind[:-1]])

                    # check cnew 
                    cnew_note = np.matmul(cnew_.T, in2_).T # frame2note
                    if np.array_equal(cnew_note, np.sign(cnew_note)) is False:
                        print(c_name, p_name)
                        raise AssertionError
                    
                    # if batch is shorter than 4 chords
                    if len(out1_) < 4: 
                        continue
                    
                    # if batch only contains rests
                    pitch = np.argmax(in1_, axis=-1)
                    uniq_pitch = np.unique(pitch)
                    if len(uniq_pitch) and uniq_pitch[0] == 88:
                        continue

                    assert in1_.shape[0] == in2_.shape[0]
                    assert in2_.shape[1] == in4_.shape[0]
                    assert in3_.shape[1] == out1_.shape[0]

                    # save batch
                    savename_x = os.path.join(savepath, '{}.{}.batch_x.{}.npy'.format(
                        c_name.lower(), p_name.lower(), ind))
                    savename_c = os.path.join(savepath, '{}.{}.batch_c.{}.npy'.format(
                        c_name.lower(), p_name.lower(), ind))
                    savename_y = os.path.join(savepath, '{}.{}.batch_y.{}.npy'.format(
                        c_name.lower(), p_name.lower(), ind))
                    savename_n = os.path.join(savepath, '{}.{}.batch_n.{}.npy'.format(
                        c_name.lower(), p_name.lower(), ind)) # roll2note mat
                    savename_m = os.path.join(savepath, '{}.{}.batch_m.{}.npy'.format(
                        c_name.lower(), p_name.lower(), ind)) # note2chord mat

                    np.save(savename_x, in1_)
                    np.save(savename_n, in2_)
                    np.save(savename_m, in3_)
                    np.save(savename_c, in4_)
                    np.save(savename_y, out1_)

                    print("saved batches for {} {} --> inp size: {} / oup size: {}      ".format(
                        c_name, p_name, in1_.shape, out1_.shape), end='\r') 
                    num += 1
                print("saved batches for {} {}".format(c_name, p_name))
    
    # np.save("unique_chord_labels_CMD.npy", np.unique(chord_list))

def save_batches_HLSD(chord_type='simple'):    
    print("Saving batches...")

    parent_path = sep.join(['.','HLSD','exp'])
    groups = sorted(glob(os.path.join(parent_path, "*"+sep)))
    maxlen, hop = 16, 8
    chord_list = list()

    for g, group in enumerate(groups): # train/val/test
        datapath = os.path.join(group, 'raw')
        savepath = os.path.join(group, "batch")
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        pieces = sorted(glob(os.path.join(datapath, 'features.*.npy')))

        for piece in pieces:
            p_name = os.path.basename(piece).split('.')[-2]
            features = np.load(piece, allow_pickle=True).tolist()            
            chords = features['chord']
            for chord in chords:
                chord_name = "{}{}_".format(chord['quality'], chord['type'])
                if chord_name not in chord_list:
                    chord_list.append(chord_name)
            # print(p_name)

            # get onehot data
            inp, oup, key, onset, onset_xml, inds = get_roll_HLSD(features, chord_type=chord_type)
            # print(p_name)
            # get indices where new chord
            new_chord_ind = [i for i, c in enumerate(onset[:,1]) if c == 1]
            new_chord_ind.append(len(inp))

            # make sure new_chord_inds are equally distanced
            prev_ind = None 
            for c in new_chord_ind: 
                if prev_ind is not None:
                    # print(c - prev_ind) 
                    assert c - prev_ind == 8
                prev_ind = c

            # save batch 
            num = 1
            for b in range(0, len(new_chord_ind), hop):

                ind = ind2str(num, 3)
                chord_ind = new_chord_ind[b:b+maxlen+1]
                start, end = chord_ind[0], chord_ind[-1] # (maxlen+1)th chord
                in1_ = inp[start:end]
                nnew_ = onset[start:end, :1]
                cnew_ = onset[start:end, -1:]
                nnew2_ = onset_xml[start:end, :1]
                cnew2_ = onset_xml[start:end, -1:]
                
                note_ind = [i for i, n in enumerate(nnew_) if n == 1]
                # roll2note 
                in2_ = make_align_matrix_roll2note(in1_, nnew_)
                # note2chord
                in3_ = make_align_matrix_note2chord(nnew_, cnew_)
                
                # get data in different units
                in4_ = np.asarray(key)
                out1_ = np.asarray([oup[c] for c in chord_ind[:-1]])

                # check cnew 
                cnew_note = np.matmul(cnew_.T, in2_).T # frame2note
                if np.array_equal(cnew_note, np.sign(cnew_note)) is False:
                    print(p_name)
                    raise AssertionError

                # if batch is shorter than 4 chords
                if len(out1_) < 4: 
                    continue
                
                # if batch only contains rests
                pitch = np.argmax(in1_, axis=-1)
                uniq_pitch = np.unique(pitch)
                if len(uniq_pitch) and uniq_pitch[0] == 88:
                    continue

                assert in1_.shape[0] == in2_.shape[0]
                assert in3_.shape[1] == out1_.shape[0]

                # save batch
                savename_x = os.path.join(savepath, '{}.batch_x.{}.npy'.format(
                    p_name.lower(), ind))
                savename_c = os.path.join(savepath, '{}.batch_c.{}.npy'.format(
                    p_name.lower(), ind))
                savename_y = os.path.join(savepath, '{}.batch_y.{}.npy'.format(
                    p_name.lower(), ind))
                savename_n = os.path.join(savepath, '{}.batch_n.{}.npy'.format(
                    p_name.lower(), ind)) # roll2note mat
                savename_m = os.path.join(savepath, '{}.batch_m.{}.npy'.format(
                    p_name.lower(), ind)) # note2chord mat

                np.save(savename_x, in1_)
                np.save(savename_n, in2_)
                np.save(savename_m, in3_)
                np.save(savename_c, in4_)
                np.save(savename_y, out1_)

                print("saved batches for {} --> inp size: {} / oup size: {}      ".format(
                    p_name, in1_.shape, out1_.shape), end='\r') 
                num += 1
            print("saved batches for {}".format(p_name))

    # np.save("unique_chord_labels_HLSD.npy", np.unique(chord_list))


def create_h5_dataset(dataset=None, setname=None): # save npy files into one hdf5 dataset
    batch_path = sep.join([".", dataset, "exp", setname, "batch"])
    files = glob(os.path.join(batch_path, "*.npy"))
    if len(files) != 0:
        # load filenames
        x1_path = [np.string_(x) for x in sorted(glob(os.path.join(batch_path, "*.batch_x.*.npy")))]
        x2_path = [np.string_(x) for x in sorted(glob(os.path.join(batch_path, "*.batch_m.*.npy")))]
        x3_path = [np.string_(x) for x in sorted(glob(os.path.join(batch_path, "*.batch_n.*.npy")))]
        y1_path = [np.string_(y) for y in sorted(glob(os.path.join(batch_path, "*.batch_y.*.npy")))]
        x4_path = [np.string_(x) for x in sorted(glob(os.path.join(batch_path, "*.batch_c.*.npy")))]

        # save h5py dataset
        f = h5py.File("{}_{}.h5".format(dataset, setname), "w")
        dt = h5py.special_dtype(vlen=str) # save data as string type
        f.create_dataset("x", data=x1_path, dtype=dt)
        f.create_dataset("m", data=x2_path, dtype=dt)
        f.create_dataset("n", data=x3_path, dtype=dt)
        f.create_dataset("c", data=x4_path, dtype=dt)
        f.create_dataset("y", data=y1_path, dtype=dt)

        f.close()

def group_notes_ind(xml_notes):
    grouped_notes = list()
    prev_note = xml_notes[0]
    grouped = [0]
    for i, note in enumerate(xml_notes[1:]):
        i += 1
        if note == prev_note:
            grouped.append(i)
        else:
            grouped_notes.append(grouped)
            grouped = [i]
        prev_note = note
    grouped_notes.append(grouped)
    return grouped_notes

def render_melody_chord_CMD(croots, ckinds, xml_notes, m, 
    save_melody=False, save_chord=False, savepath=None, save_mid=True, return_mid=False):
    
    prev_note = None
    midi_notes = list()
    melody_track = list()
    chord_track = list()
    first_onset = np.min([n['note'].note_duration.time_position for n in xml_notes])
    end_time = 0

    grouped_ind = group_notes_ind(xml_notes)
    min_oct = int(np.min([int(n['note'].pitch[0][-1]) \
        for n in xml_notes if n['note'].pitch is not None]))

    for i, note in enumerate(xml_notes):
        onset = note['note'].note_duration.time_position - first_onset
        sec = note['note'].note_duration.seconds 
        dur = note['note'].note_duration.duration
        measure_dur = note['measure'].duration
        dur2sec = sec / dur
        offset = onset + sec
        # print(onset, offset)
        if note['note'].pitch is None:
            pass 
        elif note == prev_note:
            pass
        else:
            pitch = note['note'].pitch[1]
            if min_oct <= 3:
                pitch += (12 * (4-min_oct)) 
            midi_note = pretty_midi.containers.Note(
                velocity=108, pitch=pitch, start=onset, end=offset) 
                
            midi_notes.append(midi_note)
            melody_track.append(midi_note)

        prev_note = note
    melody_offset = offset

    chord_oct = 4
    chord_sec = 1 # 4/4, 120 BPM
    notenum = np.sum(m, axis=0)
    assert len(notenum) == len(croots) == len(ckinds)
    
    for croot, ckind in zip(croots, ckinds):        
        onset = end_time
        offset = onset + chord_sec
        # print(start_time, end_time, chord_sec)
        chord_notes = get_chord_notes(ckind, croot, pc_norm=False)
        # chord_notes = [chord_notes[0]-12] + chord_notes # add base
        # print(chord_notes)

        for cnote in chord_notes:
            pitch = cnote + chord_oct * 12
            midi_cnote = pretty_midi.containers.Note(
                velocity=84, pitch=pitch, start=onset, end=offset)    
            # print(midi_cnote)
            midi_notes.append(midi_cnote) 
            chord_track.append(midi_cnote)

        end_time = offset 
    chord_offset = offset 

    assert melody_offset == chord_offset

    if save_mid is True:
        save_new_midi(
            midi_notes, ccs=None, new_midi_path=savepath, start_zero=True)

    if save_melody is True:
        save_new_midi(
            melody_track, ccs=None, new_midi_path=savepath, start_zero=True)

    if save_chord is True:
        save_new_midi(
            chord_track, ccs=None, new_midi_path=savepath, start_zero=True)

    if return_mid is True:
        return [melody_track, chord_track]

def render_melody_chord_HLSD(croots, ckinds, features, note_inds, m, 
    save_melody=False, save_chord=False, savepath=None, save_mid=True, return_mid=False):
    
    FI = FeatureIndex(dataset="HLSD")

    notes = [features['melody'][n] for n in note_inds]
    prev_note = None
    midi_notes = list()
    melody_track = list()
    chord_track = list()
    first_onset = np.min([n['start'] for n in notes])
    end_time = 0
    key_root = features['orig_key_info'][0]
    key_ind = FI.root2ind[check_chord_root(key_root)] 

    min_oct = int(np.min([(int(n['pitch'] - key_ind)//12 - 1) for n in notes if n['pitch'] > 0]))
    unit = 0.5
    for i, note in enumerate(notes):
        # 1 == one beat == 1/2 sec
        onset = (note['start'] - first_onset) * unit 
        offset = (note['end'] - first_onset) * unit
        offset = np.min([offset, len(croots)])
        # print(onset, offset)
        if note['pitch'] == 0:
            pass 
        elif note == prev_note:
            pass
        else:
            pitch = int(note['pitch']) - key_ind # normalize to C Major
            if min_oct <= 3:
                pitch += (12 * (4-min_oct)) 
            midi_note = pretty_midi.containers.Note(
                velocity=108, pitch=pitch, start=onset, end=offset) 
                
            midi_notes.append(midi_note)
            melody_track.append(midi_note)

        prev_note = note
    melody_offset = offset

    chord_oct = 4
    chord_sec = 1 # 4/4, 120 BPM
    notenum = np.sum(m, axis=0)
    assert len(notenum) == len(croots) == len(ckinds)
    
    for croot, ckind in zip(croots, ckinds):        
        onset = end_time
        offset = onset + chord_sec
        # print(start_time, end_time, chord_sec)
        chord_notes = get_chord_notes(ckind, croot, pc_norm=False)
        # chord_notes = [chord_notes[0]-12] + chord_notes # add base
        # print(chord_notes)

        for cnote in chord_notes:
            pitch = cnote + chord_oct * 12
            midi_cnote = pretty_midi.containers.Note(
                velocity=84, pitch=pitch, start=onset, end=offset)    
            # print(midi_cnote)
            midi_notes.append(midi_cnote) 
            chord_track.append(midi_cnote)

        end_time = offset 
    chord_offset = offset 

    assert melody_offset <= chord_offset, print(melody_offset, chord_offset)

    if save_mid is True:
        save_new_midi(
            midi_notes, ccs=None, new_midi_path=savepath, start_zero=True)

    if save_melody is True:
        save_new_midi(
            melody_track, ccs=None, new_midi_path=savepath, start_zero=True)

    if save_chord is True:
        save_new_midi(
            chord_track, ccs=None, new_midi_path=savepath, start_zero=True)

    if return_mid is True:
        return [melody_track, chord_track]

def render_melody_chord_Q(croots, ckinds, notes, m, 
    save_melody=False, save_chord=False, savepath=None, save_mid=True, return_mid=False):
    notenum = np.sum(m, axis=0)
    prev_note = None
    midi_notes = list()
    melody_track = list()
    chord_track = list()
    end_time = 0

    octaves = list()
    for note in notes:
        pitch = np.argmax(note, axis=0) + 21  
        octave = (pitch // 12) - 1 
        octaves.append(octave)
    min_oct = int(np.min(octaves))

    offset = 0
    for i, note in enumerate(notes):
        onset = offset
        sec = np.sum(note) * (0.5/4) # 16th
        offset = onset + sec
        # print(onset, offset)

        pitch = np.argmax(note, axis=0) + 21  
        if pitch == 109:
            pass 
        else:
            if min_oct <= 3:
                pitch += (12 * (4-min_oct)) 
            midi_note = pretty_midi.containers.Note(
                velocity=108, pitch=pitch, start=onset, end=offset) 
                
            midi_notes.append(midi_note)
            melody_track.append(midi_note)
    melody_offset = offset

    if min_oct <= 2:
        chord_oct = min_oct + (3 - min_oct) 
    else:
        chord_oct = min_oct

    chord_sec = 1 # 4/4, 120 BPM
    for each_chord, croot, ckind in zip(notenum, croots, ckinds):        
        onset = end_time
        offset = onset + chord_sec
        # print(start_time, end_time, chord_sec)
        chord_notes = get_chord_notes(ckind, croot, pc_norm=False)
        # chord_notes = [chord_notes[0]-12] + chord_notes # add base
        # print(chord_notes)

        for cnote in chord_notes:
            pitch = cnote + chord_oct * 12
            midi_cnote = pretty_midi.containers.Note(
                velocity=84, pitch=pitch, start=onset, end=offset)    
            # print(midi_cnote)
            midi_notes.append(midi_cnote) 
            chord_track.append(midi_cnote)
        end_time = offset 
    chord_offset = offset
    
    assert melody_offset == chord_offset

    if save_mid is True:
        save_new_midi(
            midi_notes, ccs=None, new_midi_path=savepath, start_zero=True)

    if save_melody is True:
        save_new_midi(
            melody_track, ccs=None, new_midi_path=savepath, start_zero=True)

    if save_chord is True:
        save_new_midi(
            chord_track, ccs=None, new_midi_path=savepath, start_zero=True)

    if return_mid is True:
        return [melody_track, chord_track]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', default=None)
    args = parser.parse_args()
    if args.dataset == 'CMD':
        print()
        print("---------- START PARSING CMD DATASET ----------")
        subprocess.call(['python', 'CMD_parser_features.py'])
        print("---------- END PARSING CMD DATASET ----------")
        print()
        print("---------- START SAVING CMD BATCHES ----------")
        split_sets_CMD()
        save_batches_CMD()
        print("---------- END SAVING CMD BATCHES ----------")
        print()
    elif args.dataset == 'HLSD':
        print()
        print("---------- START PARSING HLSD DATASET ----------")
        subprocess.call(['python', 'HLSD_parser.py'])
        subprocess.call(['python', 'HLSD_features.py'])
        print()
        print("---------- START SAVING HLSD BATCHES ----------")
        split_sets_HLSD()
        save_batches_HLSD()
        print("---------- END SAVING HLSD BATCHES ----------")
        print()

    # save dataset in h5py format 
    for s in ['train', 'val', 'test']:
        create_h5_dataset(dataset=args.dataset, setname=s)
    print("---------- SAVED H5 DATASET -> READY TO TRAIN ----------")
    print()