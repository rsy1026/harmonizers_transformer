import numpy as np
import os
import sys 
sys.path.append('/utils')
from glob import glob
import pretty_midi 
import h5py
import pandas as pd 
import shutil
from decimal import Decimal
from scipy.stats import entropy

from utils.parse_utils import *
from utils import tonal_distance
from process_data import *


def compute_metrics(_x, _y, _n, _m, _k, midi_save_path=None, dataset=None):
    '''
    Ref: 
    - Automatic Melody Harmonization with Triad Chords: A Comparative Study (Yeh et al., 2020)
    - On the evaluation of generative models in music (Yang et al., 2018)
    - A geometrical distance measure for determining the similarity of musical harmony (de Haas et al., 2016)
    - A Directional Interval Class Representation of Chord Transitions (Cambouropoulos, 2012)

    ** What we need: 
    1) [CHE, CC, CTnCTR, PCS, TPS, DIC]
    - chord output (y): generated chord (probability)
        - for calculating chord histogram, ckind and croot    
    - melody input roll (x) 
    - frame2note alignment matrix (n)
    - note2chord alignment matrix (m)

    2) [CTD, MCTD]
    - midi file
        - melody track: melody all splitted by chord boundaries
        - chord track 
    '''

    FI = FeatureIndex(dataset=dataset)

    ind2chord = FI.ind2chord_func_simple
    chord2ind = FI.chord2ind_func_simple
    ind2root = FI.ind2root

    chord_ind = _y
    croot_ind = chord_ind // 6
    ckind_ind = np.mod(chord_ind, 6)
    chord_lab = ["{}{}".format(ind2root[cr], ind2chord[ck]) \
        for cr, ck in zip(croot_ind, ckind_ind)]
    ckind_lab = ["{}".format(ind2chord[ck]) \
        for ck in ckind_ind]

    x_note_sum = np.matmul(_x.T, _n).T

    # midi track (midi objects)
    melody, chords = render_melody_chord_Q(
        croot_ind, ckind_lab, x_note_sum, _m, 
        savepath=midi_save_path, save_mid=False, return_mid=True)   

    ctd_all, mctd_all = tonal_distance.get_ctd_mtd(melody, chords)
    CTD = np.mean(ctd_all)
    MCTD = np.mean(mctd_all)

    if midi_save_path is not None:
        np.save(midi_save_path + ".npy", [melody, chords])

    ## Chord Histrogram Entropy (CHE) ## 
    # initialize chord histrogram
    chord_hist = dict()
    for i in range(72):
        chord_hist[i] = 0

    # get chord histrogram
    for c in chord_ind:
        chord_hist[c] += 1 

    # get probability 
    chord_count = np.asarray([h[1] for h in chord_hist.items()])
    CH = chord_count
    chord_prob = chord_count / chord_count.sum()

    # compute entropy
    CHE = entropy(chord_prob)
    CHE = np.round(CHE, 4)
    # print("     --> CHE: {:.4f}".format(CHE))

    ## Chord coverage (CC) ## 
    CC = np.sum(np.sign(chord_count))
    CC = np.round(CC, 4)
    # print("     --> CC: {}".format(int(CC)))


    ## Tonal Pitch Space (TPS) & Directed Interval Class (DIC) ##
    prev_cr = None
    prev_levels = None
    prev_chord_notes = None
    TPS_all = list()
    DIC_all = list()

    for ck, cr, ci in zip(ckind_lab, croot_ind, chord_ind):
        chord_notes = get_chord_notes_simple(ck, cr, pc_norm=True)
        lev_a = [cr] 
        lev_b = [cr, (cr+7)%12]
        lev_c = chord_notes
        levels = [lev_a, lev_b, lev_c]

        ## DIC list ##
        if prev_chord_notes is not None:
            DIC = np.zeros([12,])
            for p in prev_chord_notes:
                for c in chord_notes:
                    diff = c - p
                    if diff <= -6:
                        diff = diff + 12 
                    elif diff > 6:
                        diff = diff - 12 
                    # print(diff)
                    assert diff >= -5 and diff <= 6
                    ind = diff + 5 # min is -5 
                    DIC[ind] += 1
            DIC_all.append(DIC)

        ## TPS list ##
        # normalize chord root into C Major 
        cr_norm = cr - _k
        if cr_norm < 0:
            cr_norm += 12

        # compute transition matrix 
        if prev_levels is not None:
            
            # find overlaps (TPS: chord distance rule)
            overlap_a = np.intersect1d(prev_levels[0], lev_a)
            overlap_b = np.intersect1d(prev_levels[1], lev_b)
            overlap_c = np.intersect1d(prev_levels[2], lev_c)
            dp_a = len(lev_a) - len(overlap_a)
            dp_b = len(lev_b) - len(overlap_b)
            dp_c = len(lev_c) - len(overlap_c)

            # cof step (TPS: circle-of-fifth rule)
            num_cof_step = 0
            added_cof = copy.deepcopy(prev_cr)
            while added_cof != cr:
                added_cof = (added_cof+7) % 12
                # print(added_cof)
                num_cof_step += 1
            if num_cof_step > 6:
                num_cof_step = 12 - num_cof_step 

            TPS = dp_a + dp_b + dp_c + num_cof_step
            TPS_all.append(TPS)

        prev_cr = cr
        prev_levels = levels
        prev_chord_notes = chord_notes


    ## Chord tone to non-chord tone ratio (CTnCTR) ##
    x_note = x_note_sum # don't have to mean -> use argmax
    # calculate number of notes within each chord
    note2chord = np.sum(_m, axis=0).astype(np.int)

    start_note = 0
    all_pcs_num = 0
    ct_dict = {'ct': 0, 'nct': 0, 'pnct': 0}
    pcs_sum = 0
    for i, note_num in enumerate(note2chord):
        next_note = [] 
        end_note = start_note + note_num
        in_notes = x_note[start_note:end_note] # notes within a chord
        if end_note < len(x_note)-1: 
            next_note = x_note[end_note] 
        else: 
            next_note = []
        ckind = ckind_lab[i]
        croot = croot_ind[i]
        for k, note in enumerate(in_notes): # each note
            pitch = np.argmax(note, axis=-1) + 21 
            if pitch == 109:
                continue

            pc = pitch % 12
            ct, chord_notes = decide_ct_simple(ckind, croot, pc, pc_norm=True)
            
            # whether note is CT
            if ct == True: # chord_tone 
                ct_dict['ct'] += 1 
            elif ct == False: 
                # get right next note of current note
                if k < len(in_notes)-1:
                    right_next_note = in_notes[k+1] 
                else:
                    right_next_note = next_note  
                # get whether CT
                if len(right_next_note) == 0:
                    ct_dict['nct'] += 1 
                else: 
                    next_pitch = np.argmax(right_next_note, axis=0) + 21 
                    if np.abs(next_pitch - pitch) <= 2:
                        ct_dict['pnct'] += 1 
                    else:
                        ct_dict['nct'] += 1

            # compute PCS
            pcs_list = list()
            for cn in chord_notes:
                # get interval (pitch always top of chord)
                intv_raw = pc - cn 
                if intv_raw < 0:
                    intv = (intv_raw + 12) % 12
                else:
                    intv = intv_raw % 12 
                # get PCS
                if intv in [0, 3, 4, 7, 8, 9]: # unison, m/M3, m/M6, p5 
                    pcs = 1 
                elif intv == 5:
                    pcs = 0 
                else:
                    pcs = -1 
                # print(intv_raw, intv, pcs)
                pcs_list.append(pcs)
            pcs_sum += (np.sum(pcs_list) * np.sum(note)) # tile by number of frame 
            all_pcs_num += (len(pcs_list) * np.sum(note)) # sum for averaging PCSs

        # update onset
        start_note = end_note
        
    # compute metric
    CTnCTR = (ct_dict['ct'] + ct_dict['pnct']) / (ct_dict['ct'] + ct_dict['nct'])
    CTnCTR = np.round(CTnCTR, 4)
    PCS = pcs_sum / all_pcs_num
    PCS = np.round(PCS, 4)


    return [CHE, CC, CTnCTR, PCS, CTD, MCTD], [TPS_all, DIC_all]


def quantitative_metrics(inputs, dataset=None):

    x, y, n, m, k = inputs
    results1, results2 = compute_metrics(x, y, n, m, k, dataset=None)

    return results1, results2


def pad_to_longer_sequence(a, b):
    if len(a) > len(b):
        diff = len(a) - len(b)
        mask = np.concatenate([np.ones([len(b),]), np.zeros([diff,])], axis=0)
        a_ = a 
        b_ = np.concatenate([b, np.zeros([diff,])], axis=0) # pad b        
    elif len(a) < len(b):
        diff = len(b) - len(a)
        mask = np.concatenate([np.ones([len(a),]), np.zeros([diff,])], axis=0)
        a_ = np.concatenate([a, np.zeros([diff,])], axis=0) # pad a
        b_ = b 
    else:
        mask = np.ones([len(a),])
        a_, b_ = a, b # pad nothing
    return a_, b_, mask


def levenshtein(s1, s2, debug=False):
    '''
    Ref: https://lovit.github.io/nlp/2018/08/28/levenshtein_hangle/
    '''
    if len(s1) < len(s2):
        return levenshtein(s2, s1, debug)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        if debug:
            print(current_row[1:])

        previous_row = current_row

    return previous_row[-1]


def chord_similarity(model_results, GT_results):
    '''
    Compare b/t model and GT  
    '''

    TPS, DIC, out = model_results
    gTPS, gDIC, gout = GT_results 

    TPS_, gTPS_, mask = pad_to_longer_sequence(TPS, gTPS)

    ## TPS ## 
    all_areas = list()
    for s in range(len(TPS_)):
        shifted = np.roll(TPS_, s)
        area = np.sum(np.abs(shifted - gTPS_) * mask)
        all_areas.append(area)
    TPSD = np.min(all_areas) / len(TPS_)

    ## DIC ## 
    DICD = np.sum(np.abs(DIC - gDIC))

    # levenstein distance 
    LD = levenshtein(out, gout) / len(out)


    return TPSD, DICD, LD




