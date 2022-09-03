import os, sys
import numpy as np
import pretty_midi as pmd

import argparse

def init_parser():
	parser = argparse.ArgumentParser(description='Parse midi files to generate chord tonal distance and melody-chord distance.')
	parser.add_argument('--data_dir', dest='file_dir', default='./nlp4musa_testQ_data', help='a directory for input numpy data files.')
	parser.add_argument('--file_name', dest='file_name', type=str, default='all', help='file name to calculate CTD and MCTD. \
		Default is to calculate mean and standard deviation of all files.')
	return parser

def transition_matrix(r1=1.0, r2=1.0, r3=0.5):
	'''
	build transition matrix according to (Sandler, and Gasser 2006)
	'''
	pi = np.pi
	arr = []
	for l in range(12):
		arr.append([
			r1 * np.sin(l * 7 * pi / 6),
			r1 * np.cos(l * 7 * pi / 6),
			r2 * np.sin(l * 3 * pi / 2),
			r2 * np.cos(l * 3 * pi / 2),
			r3 * np.cos(l * 2 * pi / 3),
			r3 * np.cos(l * 2 * pi / 3)
		])
	return np.array(arr)

def to_onehot(chord):
	'''
	create one-hot vector from the pitch class of the note.
	'''
	onehot = []
	for i in range(12):
		if i in chord:
			onehot.append(1)
		else:
			onehot.append(0)
	return np.array(onehot)

def get_ctd_mtd(melody_notes, chord_notes):
	'''
	calculate chord tonal distances and melody-chord tonal distances of a given song(numpy file).
	return type is a tuple of chord tonal distance array and melody-chord tonal distance array.
	'''
	ctd_distances = []
	mctd_distances = []
	tm = transition_matrix()
	
	# midi_data = np.load(file_dir, allow_pickle=True)

	# melody_notes, chord_notes = midi_data[0], midi_data[1]
	chord = []
	chords = []
	start = chord_notes[0].start
	end = chord_notes[0].end
	# align chord notes with melody notes
	for note in chord_notes:
		if note.start == start:
			chord.append(note.pitch % 12)
		else:
			chords.append((start, end, chord[:]))
			start = note.start
			end = note.end
			chord = [note.pitch % 12]
	chords.append((start, end, chord[:]))
	
	chroma_vectors = list(map(lambda x: (x[0], x[1], to_onehot(x[2])), chords))
	chord_centroids = list(map(lambda x: (x[0], x[1], np.matmul(x[2], tm) / np.sum(x[2])), chroma_vectors))

	for note in melody_notes:
		note_chroma = to_onehot([note.pitch % 12])
		note_centroid = np.matmul(note_chroma, tm)
		for chord in chord_centroids:
			if note.start >= chord[0] and note.end <= chord[1]: # only applies for aligned note
				mctd_distances.append(np.linalg.norm(chord[2] - note_centroid))

	for i in range(len(chord_centroids)-1):
		ctd_distances.append(np.linalg.norm(chord_centroids[i][2] - chord_centroids[i+1][2]))

	return np.array(ctd_distances), np.array(mctd_distances)

def get_ctd_mtd_all(args):
	'''
	Calculate ctd and mtd for all files in the given directory.
	Suppose argument directory has subdirectories of different models,
	which is the actual container of numpy arrays.
	'''

	test_dirs = list(filter(lambda x: not x.endswith('.npy'), os.listdir(args.file_dir)))
	test_dirs = list(map(lambda x: args.file_dir + '/' + x, test_dirs))

	# key of the dictionary is a name of the model, and value is aggregated ctd and mtd values.
	ctd_distance_info = {}
	mctd_distance_info = {}
	for directory in test_dirs:
		midis = list(filter(lambda x: x.endswith('.npy'), os.listdir(directory)))
		midis = list(map(lambda x: directory + '/' + x, midis))
		midis_cnt = len(midis)
		ctd_distances = []
		mctd_distances = []

		for midi_idx, midi in enumerate(midis):
			if midi_idx % 100 == 99:
				print(f'{directory.split("/")[-1]} : {midi_idx + 1} / {midis_cnt}')

			# extract ctd array and mtd array from individual file
			ctd, mctd = get_ctd_mtd(midi)
			
			ctd_distances.extend(ctd)
			mctd_distances.extend(mctd)
			
		ctd_distance_info[directory] = np.array(ctd_distances)
		mctd_distance_info[directory] = np.array(mctd_distances)

	# print result
	out = ''
	out += '-'*50 + '\n'
	out += 'Chord tonal distance\n\n'
	for model in ctd_distance_info:
		out += f'{model.split("/")[-1]}\n'
		out += f'mean: {ctd_distance_info[model].mean():.4f}, std: {ctd_distance_info[model].std():.4f}\n'
		out += '\n'
	out += '-'*50 + '\n'
	out += 'Melody-Chord tonal distance\n\n'
	for model in mctd_distance_info:
		out += f'{model.split("/")[-1]}\n'
		out += f'mean: {mctd_distance_info[model].mean():.4f}, std: {mctd_distance_info[model].std():.4f}\n'
		out += '\n'
	print(out)
	with open('result.txt', 'w') as f:
		f.write(out)

if __name__ == '__main__':
	parser = init_parser()
	args = parser.parse_args()
	if args.file_name == 'all':
		get_ctd_mtd_all(args)
	else:
		get_ctd_mtd(args.file_name)