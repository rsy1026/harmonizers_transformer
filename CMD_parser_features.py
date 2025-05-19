import os
import sys
sys.setrecursionlimit(100000)
sys.path.append("./utils")
import numpy as np
from glob import glob
from fractions import Fraction
import pretty_midi
import csv

from decimal import Decimal, getcontext, ROUND_HALF_UP, InvalidOperation

from utils.musicxml_parser import MusicXMLDocument
from utils.parse_utils import *

dc = getcontext()
dc.prec = 48
dc.rounding = ROUND_HALF_UP

'''
Chord Melody Dataset
- Ref: https://github.com/shiehn/chord-melody-dataset
	- Each bar must contain:
		2 chords
		all notes MUST be monophonic (i.e single note melodies)
		notes and or rests that sum to 4 beats (4/4 time)
		triplets are NOT supported!
		the maximum resolution supported is 16th notes
** Save the original dataset at the directory 
   where this code belongs to --> /(codeDiretory)/CMD/dataset/abc/...
'''

def save_features():

	sep = os.sep
	dirname = sep.join(['.','CMD','dataset'])
	savename = sep.join(['.','CMD','output'])
	categs = sorted(glob(os.path.join(dirname, "*{}").format(sep))) # songs
	
	for categ in categs:
		c_name = categ.split(sep)[-2] # song titles
		files = sorted(glob(os.path.join(categ, "*.xml"))) # 12 keys
		for key, xml in enumerate(files):
			p_name = os.path.basename(xml).split('.')[0] # each key
			input_list = parse_CMD_features(xml, save_csv=True)
			song_dir = os.path.dirname(xml).split(sep)[-1]
			save_filename = os.path.join(savename, song_dir)
			if not os.path.exists(save_filename):
				os.makedirs(save_filename)
			# save features
			np.save(os.path.join(save_filename, 
				"features.{}.{}.npy".format(c_name, p_name)), np.asarray(input_list, dtype=object))

		print("parsed input for: {}".format(c_name))


def parse_CMD_features(xml, save_csv=False):

	XMLDocument = MusicXMLDocument(xml)
	xml_notes = extract_xml_notes(
		XMLDocument, note_only=False, apply_grace=False, apply_tie=False)

	# gather measures only
	xml_measures = list()
	prev_measure_number = -1
	for i in range(len(xml_notes)):
		xml_note = xml_notes[i]['note']
		xml_measure = xml_notes[i]['measure']

		if xml_note.measure_number > prev_measure_number:
			xml_measures.append(xml_measure)
		
		prev_measure_number = xml_note.measure_number 

	# get measure duration (assume all data has 4/4 time signature & 120 BPM)
	# make sure measure duration equals to 2 sec. (500ms * 4)
	for i in range(len(xml_measures[:-1])): 
		measure_onset = xml_measures[i].notes[0].note_duration.time_position
		next_measure_onset = xml_measures[i+1].notes[0].note_duration.time_position
		measure_dur = next_measure_onset - measure_onset
		assert measure_dur == 2.

	# PARSE FEATURES
	note_list = list()
	prev_xml_note = None
	prev_xml_measure = None
	for i in range(len(xml_notes)):
		xml_note = xml_notes[i]['note']
		xml_measure = xml_notes[i]['measure']

		# parse features for each note
		parsed_note = CMDFeatures_note(note=xml_note,
								  		 measure=xml_measure,
								  	 	 prev_note=prev_xml_note,
								  		 prev_measure=prev_xml_measure,
								  		 note_ind=i)

		_input = parsed_note._input		
		note_list.append([i, _input])
		# update previous measure number and onset time 
		prev_xml_note = parsed_note # InputFeatures object
		prev_xml_measure = parsed_note.measure # xml print-object

	measure_list = list()
	prev_xml_measure = None
	for i in range(len(xml_measures)):

		# parse features for each note
		parsed_note = CMDFeatures_measure(measure=xml_measures[i],
								  		  prev_measure=prev_xml_measure)

		_input = parsed_note._input		
		measure_list.append([i, _input])
		# update previous measure number and onset time 
		prev_xml_measure = parsed_note.measure # xml print-object

	return [note_list, measure_list]



#----------------------------------------------------------------------------------#

# Class for parsing CMD features(input/output)
class CMDFeatures_note(object):
	def __init__(self, 
				 note=None, 
				 measure=None,
				 prev_note=None,
				 prev_measure=None,
				 note_ind=None):

		# Inputs
		self.note = note # xml note
		self.measure = measure # xml measure
		self.prev_note = prev_note
		self.prev_measure = prev_measure
		self.note_ind = note_ind

		# Initialize features to parse
		if self.prev_note is None:
			self.in_measure_pos = 0
		else:
			self.in_measure_pos = self.prev_note.in_measure_pos
		self.measure_duration = self.measure.duration
		self.measure_number = self.note.measure_number
		self.time_position = None
		self.ioi = None
		self.duration = None
		self.pitch_name = None
		self.pitch_num = None
		self.mode = None
		self.key_final = None
		self.pitch_class = None
		self.octave = None
		self.pitch_norm = None 
		self.pitch_class_norm = None
		self.octave_norm = None
		self.beat = None
		self.downbeat = None
		self._input = dict()

		# get input features
		self.get_features()
		# wrap up inputs
		self.wrapup_features()

		# print("parsed {}th note(measure: {})".format(
			# self.note_ind, self.measure_number))

	def get_features(self):
		self.get_time_info() # type of duration 
		self.get_downbeat() # whether downbeat 

		if self.downbeat == 1:
			self.in_measure_pos = 0

		self.get_beat() # beat position
		# self.get_chord() # assigned chord
		self.get_pitch() # pitch class and octave
		
		# update measure position
		self.in_measure_pos += self.duration

	def wrapup_features(self):
		'''
		Make features into binary vectors
		'''
		# chord
		# self._input["chord"] = self.chord
		# key
		self._input["key"] = {
			'root': self.key_final,
			'mode': self.mode
			}		
		# pitch
		self._input["pitch_abs"] = {
			'pc': self.pitch_class, 
			'octave': self.octave,
			'num': self.pitch_num
			}
		# beat position
		self._input["downbeat"] = self.beat
		self._input["beat"] = self.beat
		self._input["measure_num"] = self.measure_number
		self._input["time_position"] = self.time_position


	def get_time_info(self):
		self.time_position = self.note.note_duration.time_position
		# duration
		self.duration = self.note.note_duration.duration
		# ioi
		if self.prev_note is None:
			self.ioi = 0 
		elif self.prev_note is not None:
			self.ioi = self.time_position - self.prev_note.time_position

	def get_pitch(self):
		'''
		get pitch from note.pitch
		'''
		# get key signature
		'''
		measure.key_signature.key --> based on fifths (major)
		- -4(Ab), ... -1(F), 0(C), 1(G), 2(D), ...
		'''
		if self.measure.key_signature is not None:
			fifths_in_measure = self.measure.key_signature.key
			if fifths_in_measure < 0: # fifth down
				key = ((fifths_in_measure * 7) % -12) + 12 # tonic pc
			elif fifths_in_measure >= 0: # fifth up
				key = (fifths_in_measure * 7) % 12 # tonic pc

			self.mode = self.measure.key_signature.mode # 'major' / 'minor'
			if self.mode == "minor":
				self.key_final = (key - 3 + 12) % 12 # minor 3 below
			elif self.mode == "major":
				self.key_final = key
			
		elif self.measure.key_signature is None:
			if self.prev_note is None:
				self.key_final = None 
				self.mode = None 
			else:
				self.key_final = self.prev_note.key_final
				self.mode = self.prev_note.mode

		if self.note.pitch is not None:
			midi_num = self.note.pitch[1]
			self.pitch_num = midi_num 
			self.pitch_name = self.note.pitch[0]
			self.pitch_class = np.mod(midi_num, 12) # pitch class
			self.octave = int(midi_num / 12) - 1 # octave
			assert self.pitch_class != None
			assert self.octave != None

	def get_downbeat(self):
		'''
		get measure number of each note and see transition point 
		- notes in same onset group are considered as one event 
		'''
		if self.prev_note is None:
			self.downbeat = 1
		elif self.prev_note is not None:
			# if in different onset group
			if self.prev_note.time_position != self.time_position:
				# new measure
				if self.measure_number != self.prev_note.measure_number: 
					self.downbeat = 1
				# same measure
				elif self.measure_number == self.prev_note.measure_number: 
					self.downbeat = 0
			# if in same onset group
			elif self.prev_note.time_position == self.time_position:
				self.downbeat = self.prev_note.downbeat

	def get_beat(self):
		'''
		get beat position within a measure with note duration in xml 
		'''
		if self.prev_note is None:
			self.beat = 0 # first beat
		elif self.prev_note is not None:
			# if in different onset group
			beat_unit = self.measure_duration // 4 # always 4/4 
			self.beat = self.in_measure_pos // beat_unit 

class CMDFeatures_measure(object):
	def __init__(self,
				 measure=None,
				 prev_measure=None):

		# Inputs
		self.measure = measure # xml measure
		self.prev_measure = prev_measure

		# Initialize features to parse
		self.measure_duration = 2. # 4/4, 120 BPM
		self.measure_number = self.measure.notes[0].measure_number
		self.time_position = self.measure.notes[0].note_duration.time_position
		self.chord = dict()
		self.chord_info = None 
		self.beat = dict()
		self._input = dict()

		# get input features
		self.get_features()
		# wrap up inputs
		self.wrapup_features()

		# print("parsed {}th note(measure: {})".format(
			# self.note_ind, self.measure_number))

	def get_features(self):
		self.get_beat() # beat position
		self.get_chord() # assigned chord

	def wrapup_features(self):
		'''
		Make features into binary vectors
		'''
		# chord
		self._input["chord"] = self.chord
		# beat position
		self._input["beat"] = self.beat
		self._input["measure_num"] = self.measure_number
		self._input["time_position"] = self.time_position

	def get_chord(self):
		'''
		get chord from measure.chord_symbols 
		- measure.chord_symbols[].kind --> m, M7, d etc.
		- measure.chord_symbols[].root --> C, D#, F etc.
		- measure.chord_symbols[].degrees --> additional chord label e.g. D#m7(b5) 
		'''

		chord_candidates = list()

		# parse dynamics within current directions 
		if len(self.measure.chord_symbols) > 0:

			for chord_symbol in self.measure.chord_symbols:

				_kind = chord_symbol.kind 
				_root = chord_symbol.root
				_degrees = chord_symbol.degrees
				chord_time = chord_symbol.time_position

				try:
					printed = chord_symbol.xml_harmony.attrib['print-object']
					if printed == 'no':
						continue
				except:
					pass

				if len(_degrees) == 0:
					_degree = '' 
				elif len(_degrees) > 0:
					_degree = _degrees[0]

				chord_candidates.append(
					{'kind': _kind, 'root': _root, 
					'degrees': _degree, 'time': chord_time})

			if len(chord_candidates) > 2:
				print()
				print("**chords in measure are more than two!")
				print("	number of chord: {}".format(len(chord_candidates))) 	
				raise AssertionError			

			if len(chord_candidates) <= 2 and len(chord_candidates) > 0:
				current_chord = chord_candidates # put all 2 chords

			elif len(chord_candidates) == 0:
				print()
				print("**no chord found!")
				raise AssertionError

			self.chord = current_chord	

		elif len(self.measure.chord_symbols) == 0:
			self.chord = [{'kind': 'none', 'root': 'none', 
				'degrees': 'none', 'time': 'none'}]		

	def get_beat(self):
		'''
		get beat position within a measure with note duration in xml 
		'''
		beat_unit = self.measure_duration / 4 # always 4/4 
		self.beat = list()
		for i in range(4):
			self.beat.append(
				{'time': self.time_position + beat_unit*i, 
				 'index': i})
		 



if __name__ == '__main__':
    save_features()
