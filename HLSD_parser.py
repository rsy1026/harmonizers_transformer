import json
import os, sys
import random
import argparse
from glob import glob

'''
Hook Lead Sheet Dataset
- Ref: https://github.com/wayne391/lead-sheet-dataset

** Save the original dataset at the directory 
   where this code belongs to --> /(codeDiretory)/HLSD/dataset/event/a/...
'''
file_path = './HLSD/dataset/event/'
result_path = './HLSD/output/'

def remove_ds(l):
    ds = '.DS_Store'
    if ds in l:
        l.remove(ds)
    return l

def replace(path):
    return result_path + '/'.join(path.split('/')[3:])

def parse_with_args(args):
    with open(args.file_dir, "r") as fr:
        illegal = False
        not_fit_cnt = 0

        dct = json.load(fr)
        aligned_dct = {}

        bim = dct['metadata']['beats_in_measure']
        if bim != '4':
            print('parse failed: the song does not have 4 beat time signature.')
            return
        if len(dct['tracks']['melody']) == 0 or len(dct['tracks']['chord']) == 0:
            print('parse failed: the song has either empty melody or empty chord.')
            return

        aligned_dct['time_signature'] = dct['metadata']['beats_in_measure']
        aligned_dct['key_signature'] = dct['metadata']['key']
        mode = dct['metadata']['mode']
        if mode == '1':
            aligned_dct['mode'] = 'major'
        elif mode == '6':
            aligned_dct['mode'] = 'minor'
        else:
            print(f'{args.file_dir}: mode {mode}')
            aligned_dct['mode'] = 'unknown'

        melody = dct['tracks']['melody']
        aligned_melodies = []
            
        prev_end = 0.0
        # initialize aligned melody dictionary
        for idx, m in enumerate(melody):
            if m is None:
                continue
            aligned_m = {}
            if m['event_on'] != prev_end:
                aligned_m['pitch'] = 0
                aligned_m['is_rest'] = True
                aligned_m['start'] = prev_end
                aligned_m['end'] = m['event_on']
                aligned_melodies.append(aligned_m)
                aligned_m = {}
            aligned_m['pitch'] = m['pitch'] + 48
            aligned_m['is_rest'] = m['isRest']
            aligned_m['start'] = m['event_on']
            aligned_m['end'] = m['event_off']
            prev_end = m['event_off']

            aligned_melodies.append(aligned_m.copy())

        aligned_dct['melody'] = aligned_melodies

        chord = dct['tracks']['chord']
        aligned_chords = []

        empty_chord = 0

        beat_cnt = 4 * dct['num_measures']
        for i in range(int(beat_cnt)):
            start, end = i, i+1
            fit = False
            aligned_c = {}
            aligned_c['start'] = start
            aligned_c['end'] = end
            for c in chord:
                if c is None:
                    continue
                if c['event_on'] <= start and end <= c['event_off']:
                    fit = True
                    aligned_c['composition'] = c['composition']
                    break
            if not fit:
                not_fit_cnt += 1
                aligned_c_list = []
                for c in chord:
                    if c is None:
                        continue
                    if start <= c['event_on'] <= end or start <= c['event_off'] <= end:
                        aligned_c_list.append(c['composition'])
                if len(aligned_c_list) == 0:
                    empty_chord += 1
                    continue
                aligned_c['composition'] = random.choice(aligned_c_list)
            aligned_c['composition'] = sorted(list(map(lambda x: x % 12, aligned_c['composition'])))
            aligned_chords.append((aligned_c.copy()))
        
        if empty_chord != 0:
            illegal = True
        aligned_dct['chord'] = aligned_chords

        # after parsing and aligning
        if not illegal:
            if not os.path.isdir('/'.join((args.out_dir + '/' + args.file_dir).split('/')[:-1])):
                os.makedirs('/'.join((args.out_dir + '/' + args.file_dir).split('/')[:-1]))
            with open(args.out_dir + '/' + args.file_dir, 'w') as fw:
                json.dump(aligned_dct, fw)
                print(f'{args.file_dir} successfully parsed at {args.out_dir}')

def parse(file_dir):
    class Args:
        def __init__(self, fd):
            self.file_dir = fd
            self.out_dir = 'output'
    args = Args(file_dir)
    parse_with_args(args)

def run_all():
    # primary subdirectories of HTD/event is alphabetical indices, followed by names of artists.
    # alpha_index = list(map(lambda x: chr(x+ord('a')), range(26)))
    # alpha_file_path = list(map(lambda x: file_path+x+'/', alpha_index))
    alpha_file_path = sorted(glob(os.path.join(file_path, '*/')))

    # number of total songs
    total_cnt = 0
    # number of successfully parsed songs
    file_cnt = 0
    # number of songs without 4 beat time signatures (not parsed)
    not_four_cnt = 0
    # number of songs with empty melody or empty chords (not parsed)
    empty_mc_cnt = 0
    # number of songs with empty chords (not parsed)
    illegal_cnt = 0
    # number of songs with unaligned chord for melody (not parsed)
    unaligned_songs = 0

    major_cnt, minor_cnt, unknown_cnt = 0, 0, 0

    for al in alpha_file_path:
        if not os.path.exists(replace(al)):
            os.makedirs(replace(al))
        artists = remove_ds(os.listdir(al))
        artist_path = list(map(lambda x: al + x + '/', artists))
        for artist in artist_path:
            if not os.path.exists(replace(artist)):
                os.makedirs(replace(artist))
            songs = remove_ds(os.listdir(artist))
            song_path = list(map(lambda x: artist + x + '/', songs))
            for song in song_path:
                if not os.path.exists(replace(song)):
                    os.makedirs(replace(song))
                files = remove_ds(os.listdir(song))
                files = list(filter(lambda x: x.endswith('_key.json'), files))
                for f in files:
                    total_cnt += 1
                    with open(song + f, "r") as fr:
                        illegal = False
                        not_fit_cnt = 0

                        dct = json.load(fr)
                        aligned_dct = {}

                        bim = dct['metadata']['beats_in_measure']
                        if bim != '4':
                            not_four_cnt += 1
                            continue
                        if len(dct['tracks']['melody']) == 0 or len(dct['tracks']['chord']) == 0:
                            empty_mc_cnt += 1
                            continue
        
                        aligned_dct['time_signature'] = dct['metadata']['beats_in_measure']
                        aligned_dct['key_signature'] = dct['metadata']['key']
                        mode = dct['metadata']['mode']
                        if mode == '1':
                            aligned_dct['mode'] = 'major'
                            major_cnt += 1
                        elif mode == '6':
                            aligned_dct['mode'] = 'minor'
                            minor_cnt += 1
                        else:
                            print(f'{song + f}: mode {mode}')
                            aligned_dct['mode'] = 'unknown'
                            unknown_cnt += 1

                        melody = dct['tracks']['melody']
                        aligned_melodies = []
                            
                        prev_end = 0.0
                        # initialize aligned melody dictionary
                        for idx, m in enumerate(melody):
                            if m is None:
                                continue
                            aligned_m = {}
                            if m['event_on'] != prev_end:
                                aligned_m['pitch'] = 0
                                aligned_m['is_rest'] = True
                                aligned_m['start'] = prev_end
                                aligned_m['end'] = m['event_on']
                                aligned_melodies.append(aligned_m)
                                aligned_m = {}
                            aligned_m['pitch'] = m['pitch'] + 48
                            aligned_m['is_rest'] = m['isRest']
                            aligned_m['start'] = m['event_on']
                            aligned_m['end'] = m['event_off']
                            prev_end = m['event_off']

                            aligned_melodies.append(aligned_m.copy())

                        aligned_dct['melody'] = aligned_melodies

                        chord = dct['tracks']['chord']
                        aligned_chords = []

                        empty_chord = 0

                        beat_cnt = 4 * dct['num_measures']
                        for i in range(int(beat_cnt)):
                            start, end = i, i+1
                            fit = False
                            aligned_c = {}
                            aligned_c['start'] = start
                            aligned_c['end'] = end
                            for c in chord:
                                if c is None:
                                    continue
                                if c['event_on'] <= start and end <= c['event_off']:
                                    fit = True
                                    aligned_c['composition'] = c['composition']
                                    break
                            if not fit:
                                not_fit_cnt += 1
                                aligned_c_list = []
                                for c in chord:
                                    if c is None:
                                        continue
                                    if start <= c['event_on'] <= end or start <= c['event_off'] <= end:
                                        aligned_c_list.append(c['composition'])
                                if len(aligned_c_list) == 0:
                                    empty_chord += 1
                                    continue
                                aligned_c['composition'] = random.choice(aligned_c_list)
                            aligned_c['composition'] = sorted(list(map(lambda x: x % 12, aligned_c['composition'])))
                            aligned_chords.append((aligned_c.copy()))
                        
                        if empty_chord != 0:
                            illegal_cnt += 1
                            illegal = True
                        if not illegal and not_fit_cnt != 0:
                            unaligned_songs += 1
                        aligned_dct['chord'] = aligned_chords


                        # after parsing and aligning
                        if not illegal:
                            with open(replace(song+f), 'w') as fw:
                                json.dump(aligned_dct, fw)
                                file_cnt += 1
                                if file_cnt % 100 == 0:
                                    print(f'{file_cnt} files parsed')
    # remove empty directories recursively
    def remove_r(path):
        if not os.path.isdir(path):
            return
        files = os.listdir(path)
        if len(files):
            for f in files:
                fullpath = os.path.join(path, f)
                if os.path.isdir(fullpath):
                    remove_r(fullpath)

        files = os.listdir(path)
        if len(files) == 0:
            os.rmdir(path)

    remove_r(result_path)
    print(f'total songs: {total_cnt}')
    print(f'songs without 4 beat time signature: {not_four_cnt}')
    print(f'songs with empty melody or empty chord: {empty_mc_cnt}')
    print(f'songs with empty chord: {illegal_cnt}')
    print('-'*50)
    print(f'parsed songs: {file_cnt}')
    print(f'songs with unaligned beats: {unaligned_songs}')
    print(f'number of mode: {major_cnt} majors, {minor_cnt} minors, {unknown_cnt} unknown')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', dest='file_dir', default='all')
    parser.add_argument('--out_dir', dest='out_dir', default='output')
    args = parser.parse_args()
    if args.file_dir == 'all':
        run_all()
    else:
        parse_with_args(args)
