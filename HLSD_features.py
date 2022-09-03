import os
import sys
sys.setrecursionlimit(100000)
sys.path.append("./utils")
import numpy as np
from glob import glob
from fractions import Fraction
import pretty_midi
import csv
import json
import copy

from decimal import Decimal, getcontext, ROUND_HALF_UP, InvalidOperation

from utils.parse_utils import *
from utils.musicxml_parser import MusicXMLDocument

dc = getcontext()
dc.prec = 48
dc.rounding = ROUND_HALF_UP

'''
Hook Lead Sheet Dataset
- Ref: https://github.com/wayne391/lead-sheet-dataset
	- Each bar must contain:
		2 chords
		all notes MUST be monophonic (i.e single note melodies)
		notes and or rests that sum to 4 beats (4/4 time)
		triplets are NOT supported!
		the maximum resolution supported is 16th notes

- First parse with "HLSD_parser.py"
	- command -> python3 HLSD_parser.py 
** Save the original dataset at the directory 
   where this code belongs to --> /(codeDiretory)/HLSD/dataset/event/a/...  
'''

sep = os.sep
dirname = sep.join(['.','HLSD','output','event'])
orig_dirname = sep.join(['.','HLSD','dataset','event'])

def save_features():
	categs = sorted(glob(os.path.join(dirname, "*{}".format(sep))))
	parsed_event = 0
	empty_event = 0
	error_json = 0

	for categ in categs:			
		pieces = sorted(glob(os.path.join(categ, "*{}".format(sep))))
		for piece in pieces:
			songs = sorted(glob(os.path.join(piece, "*{}".format(sep))))
			for song in songs:
				parts = sorted(glob(os.path.join(song, "*.json")))
				for part in parts:
					c_name = os.path.basename(categ[:-1]) # alphabet of artist name
					p_name = os.path.basename(piece[:-1]) # artist name
					s_name = os.path.basename(song[:-1]) # song title
					part_name = os.path.basename(".".join(part.split(".")[:-1]))

					try:
						input_list = parse_HLSD_features(part)
						if input_list is None:
							empty_event += 1
							print("		>> Empty melody or chord! -> {}/{}/{}".format(p_name, s_name, part_name))
							continue
						parsed_event += 1
					except json.JSONDecodeError:
						error_json += 1
						print("		>> Error in JSON file! -> {}/{}/{}".format(p_name, s_name, part_name))
						continue
					except PermissionError:
						os.chmod(part, 0o444)

					# save features
					np.save(os.path.join(os.path.dirname(part), 
						"features.{}.{}.{}.npy".format(p_name, s_name, part_name)), input_list)
						
					print("parsed {}/{}/{} condition input".format(p_name, s_name, part_name))
	print("> PARSED TOTAL {} FILES".format(parsed_event))

def parse_HLSD_features(json_file):

	'''
	Parsed by HS: 
	- all 4/4 
	- Key: Major/Minor/Others
		- should normalize others to one of M/m
	- should additionally get symbol, root, type, mode  
	'''
	# load json
	with open(json_file, "r") as st_json:
		info = json.load(st_json)

	orig_parent_path = orig_dirname
	part_path = sep.join(json_file.split(sep)[4:])
	orig_json_file = os.path.join(orig_parent_path, part_path)
	assert os.path.exists(orig_json_file)

	with open(orig_json_file, "r") as st_json2:
		orig_info = json.load(st_json2)

	melody = info['melody']
	chord = info['chord']
	mode, key = orig_info['metadata']['mode'], orig_info['metadata']['key']
	info['orig_key_info'] = [key, mode]
	orig_melody = [m for m in orig_info['tracks']['melody'] if m is not None]
	orig_chord = [c for c in orig_info['tracks']['chord'] if c is not None]
	
	if len(orig_melody) > 0 and len(orig_chord) > 0:
	
		assert melody[-1]['end'] == orig_melody[-1]['event_off']
		# assert chord[-1]['end'] == orig_chord[-1]['event_off']
		song_end_ = np.max([melody[-1]['end'], chord[-1]['end']])

		if song_end_ % 4 > 1:
			song_end = (song_end_//4+1) * 4
			print("changed song end from {} to {}".format(song_end_, song_end))
		elif song_end_ % 4 == 0:
			song_end = song_end_ 
			
		## Re-parse chords to sample 2 chords for every half-measure ## 
		'''
		Ref: https://github.com/wayne391/lead-sheet-dataset/blob/f6b72f9cf17d8010d5e55506d800e05f1790e2d6/src/tab_parser.py#L41
		-> each integer in "event_on"/"event_off" correspond to beat number 
		-> 16 measures in 4/4 ends at "64.0" (16 * 4)
		'''
		new_chord_list = list()
		prev_start = 0
		for each_chord in orig_chord:
			# original info 
			start, end = each_chord['event_on'], each_chord['event_off']
			dur = each_chord['event_duration']
			assert end - start == dur
			# group chords by every 2 beats
			if start % 2 == 0 and dur <= 2:
				# print(start, end)
				new_chord_list.append(each_chord) # w/o change
			else:
				# print(start, end, dur)
				# print("--------------------")
				new_start = start 
				new_end = None
				if start % 2 == 0 and dur > 2:
					new_end = start + 2 # new_end - new_start == 2
				elif start % 2 > 0: # no matter the duration
					new_end = (start//2)*2 + 2 # new_end - new_start < 2
				while new_end < end:
					new_chord = copy.deepcopy(each_chord) # copy original chord
					new_chord['event_on'] = new_start 
					new_chord['event_off'] = new_end
					# print(new_start, new_end)
					new_chord_list.append(new_chord)
					new_start = new_end  
					new_end = new_start+2
				# update for original end
				new_chord = copy.deepcopy(each_chord) # copy original chord
				new_chord['event_on'] = new_start 
				new_chord['event_off'] = end # original end
				# print(new_start, end)
				# print("--------------------")
				new_chord_list.append(new_chord)

		# group chords by half-measure
		half_measure_dict = dict()
		prev_half_measure = None
		beat_sum = 0
		for i in range(0, int(song_end), 2):
			half_start, half_end = i, i+2
			half_measure = list()
			# print("------------")
			# print(i, i+2)
			for each_chord in new_chord_list:
				start, end = each_chord['event_on'], each_chord['event_off']
				dur = each_chord['event_duration']
				if start >= half_start and end <= half_end:
					# print(start, end)
					half_measure.append(each_chord)
			if len(half_measure) == 0:
				assert prev_half_measure is not None
				half_measure = prev_half_measure
			half_measure_dict["{}_{}".format(half_start, half_end)] = half_measure
			prev_half_measure = half_measure

		# sample 1 chord for each half-measure
		new_half_measure_list = list()
		for mm in half_measure_dict:
			'''
			chord type: triad(5) or 7 or others *
			chord mode: mode of chord(major, minor or other; diff from key mode) 
			chord root: root * 
			chord quality: maj, min, dim, dominant *
			chord symbol: root + quality + type + etc. (overall) *
			'''
			half_start, half_end = mm.split("_")
			chord_hist = dict()
			each_half = half_measure_dict[mm]
			for h in each_half:
				chord_name = '_'.join(
					[h['symbol'], str(h['chord_type']), str(h['root']), h['quality']])
				try:
					chord_hist[chord_name] += 1
				except KeyError:
					chord_hist[chord_name] = 1
			chord_items = np.asarray([[k, v/len(each_half)] for k, v in chord_hist.items()])
			chord_s = np.random.multinomial(1, pvals=chord_items[:,1].tolist(), size=1)
			chord_ind = np.argmax(chord_s)
			picked = chord_items[chord_ind][0]
			symbol, type_, root, qual = picked.split("_")
			
			# update new chord list
			new_half_measure_list.append({
				'start': int(half_start),
				'end': int(half_end),
				'root': int(root),
				'type': type_,
				'quality': qual,
				'symbol': symbol})
		
		info['chord'] = new_half_measure_list

	else:
		info = None 

	return info



if __name__ == '__main__':
    save_features()
