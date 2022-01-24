import numpy as np
import os
import sys 
sys.path.append("./utils")
sys.path.append("./models")
import time
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F 
import importlib
from scipy.stats import truncnorm

from CMD_parser_features import parse_CMD_features
from process_data import *
from utils.parse_utils import *
from models import STHarm, VTHarm


sep = os.sep 

class TruncatedNorm(nn.Module):
    def __init__(self):
        super(TruncatedNorm, self).__init__()

    def forward(self, size, threshold=2.):
        values = truncnorm.rvs(-threshold, threshold, size=size)
        return values.astype('float32')

class TestData(object):
    def __init__(self, 
                 dataset=None,
                 song_ind=None,
                 start_point=None,
                 maxlen=16):
        super(TestData, self).__init__()

        self.test_parent_path = None 
        self.features = None 
        self.data = None
        self.test_name = None 
        sep = os.sep 

        # output 
        self.test_batches = None
        self.m2 = None 
        self.test_notes = None 
        self.key_sig = None

        if dataset == "CMD":
            self.test_parent_path = sep.join([".","CMD","exp","test","raw"])
            test_song_lists = [p.split("/")[-2] \
                for p in sorted(glob(os.path.join(self.test_parent_path, "*/")))]
            test_song = test_song_lists[song_ind]
            self.test_name = test_song
            data_parent_path = sep.join([".","CMD","dataset"])
            xml = sorted(glob(os.path.join(data_parent_path, test_song, '*.xml')))[0]
            self.features = parse_CMD_features(xml)
            xmlDoc = MusicXMLDocument(xml)
            xml_notes = extract_xml_notes(
                xmlDoc, note_only=False, apply_grace=False, apply_tie=False)
            self.CMD_data(self.features, xml_notes, start_point, maxlen) 


    def __call__(self):
        return self.test_batches, self.m2, self.test_notes

    def CMD_data(self, features, xml_notes, start_point, maxlen):

        # get onehot data
        inp, oup, key, beat, onset, onset_xml, inds = get_roll_CMD(features, chord_type="simple")

        # get indices where new chord
        new_chord_ind = list()
        for i, c in enumerate(onset[:,1]):
            if c == 1:
                new_chord_ind.append(i)
        new_chord_ind.append(len(inp))
        
        chord_ind = new_chord_ind[start_point:start_point+maxlen+1]
        start, end = chord_ind[0], chord_ind[-1] # (maxlen+1)th chord

        note_inds = list()
        for i in inds:
            if i[1] >= start and i[1] < end: 
                note_inds.append(i[0])

        _x = inp[start:end]
        key_ = key[start:end]
        beat_ = beat[start:end]
        nnew_ = onset[start:end, :1]
        cnew_ = onset[start:end, -1:]
        nnew2_ = onset_xml[start:end, :1]
        cnew2_ = onset_xml[start:end, -1:]

        _n = make_align_matrix_roll2note(_x, nnew_) # roll2note 
        _m = make_align_matrix_note2chord(nnew_, cnew_) # note2chord
        _m2 = make_align_matrix_note2chord(nnew2_, cnew2_)
        _y = np.asarray([oup[c] for c in chord_ind[:-1]]) 

        test_notes = [xml_notes[i] for i in note_inds]
        x_pitch = np.argmax(_x, axis=-1) 
        y_chord = np.argmax(_y, axis=-1) # labels
        self.key_sig = np.asarray(np.argmax(key_[0], axis=-1))
        # _clab = np.asarray(len(np.unique(y_chord)) - 1) # number of chords

        test_x = x_pitch
        test_k = self.key_sig
        test_m = _m
        test_n = _n
        test_y = y_chord

        self.test_batches = [test_x, test_k, test_m, test_n, test_y]
        self.m2 = _m2 
        self.test_notes = test_notes 


def test_model(dataset="CMD", 
               song_ind=None, 
               exp_name=None,
               device_num=None,
               lamb=None, 
               start_point=None,
               maxlen=16):

    if exp_name == "STHarm":
        model_name = exp_name 
    elif exp_name in ["VTHarm", "rVTHarm"]:
        model_name = "VTHarm"

    ## LOAD DATA ##
    test_data = TestData(dataset=dataset,
                         song_ind=song_ind,
                         start_point=start_point,
                         maxlen=maxlen)
    
    test_batch, _m2, test_notes = test_data()
    features = test_data.features
    key_sig = test_data.key_sig
    test_name = test_data.test_name

    ## LOAD MODEL ##
    module_name = model_name
    model = importlib.import_module(module_name)
    Generator = model.Harmonizer
    Mask = model.Mask
    Compress = model.Compress

    model_path = "./trained/{}_{}".format(exp_name, dataset)
    device = torch.device("cuda:{}".format(device_num))
    if torch.cuda.is_available():
        torch.cuda.set_device(device_num) 

    # select model
    model = Generator(hidden=256, n_layers=4, device=device)
    model.to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict']) 

    ## SAMPLE ##
    start_time = time.time()
    model.eval()
    test_x, test_k, test_m, test_n, test_y = test_batch
    test_x_ = torch.from_numpy(test_x.astype(np.int64)).to(device).unsqueeze(0)
    test_k_ = torch.from_numpy(test_k.astype(np.int64)).to(device).unsqueeze(0)
    test_m_ = torch.from_numpy(test_m.astype(np.float32)).to(device).unsqueeze(0)
    test_n_ = torch.from_numpy(test_n.astype(np.float32)).to(device).unsqueeze(0)
    test_y_ = torch.from_numpy(test_y.astype(np.int64)).to(device).unsqueeze(0)    
    
    if exp_name == "STHarm":   
        chord_, kq_attn_ = model.test(test_x_, test_n_, test_m_)

    elif exp_name == "VTHarm":   
        # c_sampled = torch.randn_like(c)
        trunc = TruncatedNorm()
        c_sampled = torch.FloatTensor(trunc([1, 16], threshold=3.)).to(device)
        chord_, kq_attn_ = model.test(test_x_, test_k_, test_n_, test_m_, c=c_sampled)

    elif exp_name == "rVTHarm":   
        # c_sampled = torch.randn_like(c)
        trunc = TruncatedNorm()
        c_sampled = torch.FloatTensor(trunc([1, 16], threshold=3.)).to(device)
        c_sampled[:,0] = lamb
        chord_, kq_attn_ = model.test(test_x_, test_k_, test_n_, test_m_, c=c_sampled)

    ## RESULT ##
    x_ = F.one_hot(test_x_, num_classes=89).cpu().data.numpy()[0] # onehot
    k_ = test_k
    y_ = test_y
    m_ = test_m 
    n_ = test_n 
    mask_ = Mask().seq_mask(test_n_.transpose(1, 2))[0].cpu().detach().numpy()
    FI = FeatureIndex(dataset=dataset)

    ind2chord = FI.ind2chord_func_simple
    ind2root = FI.ind2root

    y_chord_ind = y_
    y_croot_ind = y_chord_ind // 6
    y_ckind_ind = np.mod(y_chord_ind, 6)
    y_chord_lab = ["{}{}".format(ind2root[cr], ind2chord[ck]) \
        for cr, ck in zip(y_croot_ind, y_ckind_ind)]
    y_ckind_lab = ["{}".format(ind2chord[ck]) \
        for ck in y_ckind_ind]

    test_chord = F.log_softmax(chord_[0], dim=-1).cpu().detach().numpy()
    test_chord_ind = np.argmax(test_chord, axis=-1)
    test_croot_ind = test_chord_ind // 6
    test_ckind_ind = np.mod(test_chord_ind, 6)
    test_chord_lab = ["{}{}".format(ind2root[cr], ind2chord[ck]) \
        for cr, ck in zip(test_croot_ind, test_ckind_ind)]
    test_ckind_lab = ["{}".format(ind2chord[ck]) \
        for ck in test_ckind_ind]


    # render into MIDI 
    if dataset == "CMD":
        render_melody_chord_CMD(y_croot_ind, y_ckind_lab, test_notes, _m2, 
            savepath="GT__{}__s{}_p{}-{}.mid".format(
                test_name, song_ind, start_point, start_point+maxlen-1,))
        render_melody_chord_CMD(test_croot_ind, test_ckind_lab, test_notes, _m2, 
            savepath="Sampled__{}__s{}_{}_p{}-{}.mid".format(
                test_name, song_ind, exp_name, start_point, start_point+maxlen-1))




if __name__ == "__main__":
    '''
    Ex) python3 test.py CMD 0 0 rVTHarm 3 (3)
    '''
    dataset = sys.argv[1]
    song_ind = int(sys.argv[2])
    start_point = int(sys.argv[3])
    exp_name = sys.argv[4]
    device_num = int(sys.argv[5])
    try:
        lamb = int(sys.argv[6])
    except IndexError:
        lamb = None

    test_model(
        dataset=dataset, song_ind=song_ind, start_point=start_point, 
        exp_name=exp_name, device_num=device_num, lamb=lamb, maxlen=16)