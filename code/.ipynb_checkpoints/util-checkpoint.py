import evaluation as ev
import os
import psutil
import ast
import re
import numpy as np
import copy
import pandas as pd

from scipy.spatial import distance
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_similarity_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


# check usage
def usage():
    process = psutil.Process(os.getpid())
    return process.memory_info()[0] / float(2 ** 20)


# get idx of sample in each cluster
def get_idx_for_cluster(kmeans_labels, n_clusters):
    indices = []
    for c in range(n_clusters):
        idx = np.where(kmeans_labels == c)[0]
        indices.append(idx.tolist())
    return indices


# save cluster idx info
def save_cluster_idx(indices, n_clusters, fn):
    with open(fn, 'w') as f:
        f.write(str(n_clusters))
        f.write('\n')
        for idx in indices:
            f.write(str(idx))
            f.write('\n')

# load cluster idx from file
def load_cluster_idx(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
        n_clusters = lines[0].strip()
        indices = [ast.literal_eval(line.strip()) for line in lines[1 :]]
    return indices, n_clusters


# save dimension-reduced feature matrix
def save_pca(temp_filename, X_pca):
    np.savez_compressed(temp_filename, X_pca=X_pca)
    
    
def load_pca(temp_filename):
    loaded = np.load(temp_filename + '.npz')
    return loaded['X_pca']
    
# check math kernel
def check_mkl():
	np.show_config()


# load data file with indices, return raw data in list of strings   
def load_file_with_indices(data_dir, dataset_name, indices):
	with open(data_dir) as f:
		data_stats =  [int(item) for item in f.readline().strip().split()]
		data = f.read().splitlines()
		select_data = [data[i] for i in indices]
	tup = data_stats[0 : 3]
	print('Dataset={0}, num_ents={1[0]}, num_feats={1[1]}, num_labels={1[2]}'.format(dataset_name, tup))
	return select_data, tup


# load data file, return raw data in list of strings
def load_file(data_dir, dataset_name):
	with open(data_dir) as f:
		data_stats =  [int(item) for item in f.readline().strip().split()]
		data = f.read().splitlines()
	tup = data_stats[0 : 3]
	print('Dataset={0}, num_ents={1[0]}, num_feats={1[1]}, num_labels={1[2]}'.format(dataset_name, tup))
	return data, tup


# load data, parse string into feats numbers
def load_data(data):
	
	def parse_feats(feats_str):
		temp = '{'+', '.join(feats_str.split(' '))+'}'
		return ast.literal_eval(temp)
	
	def parse_labels(labels_str):
		return [int(l) for l in labels_str.split(',')]

	pat_both = re.compile(r'((\d+,{0,1})+) ((\d+:\d+.\d* {0,1})+)')
	pat_feats = re.compile(r'(( (\d+:\d+.\d* {0,1})+))')
	
	feats_labels_list = []
	
	avg_feats = 0
	avg_labels = 0
	
	for ent in data:
		labels = []
		if re.match(pat_both, ent):
			labels = parse_labels(re.match(pat_both, ent).group(1))
			feats = parse_feats(re.match(pat_both, ent).group(3))
		else:
			feats = parse_feats(re.match(pat_feats, ent).group()[1 :])
		feats_labels_list.append((feats, labels))
		
		avg_feats += len(feats)
		avg_labels += len(labels)
	
	return feats_labels_list, avg_feats / len(feats_labels_list), avg_labels / len(feats_labels_list)


# generate X and Y
def gen_data(feats_labels_list, spec_tup, normalized_x=True, normalized_y=True):
	
	num_samples = len(feats_labels_list)
	X = np.zeros((num_samples, spec_tup[1]), dtype='float32')
	Y = np.zeros((num_samples, spec_tup[2]), dtype='float32')
	
	for i, (feats, labels) in enumerate(feats_labels_list[:]):
		#print(feats.items())
		X[i, list(feats.keys())] += list(feats.values())
		Y[i, labels] += 1
        
	if normalized_x:
		X = np.multiply((1.0 / np.linalg.norm(X, axis=1))[:, np.newaxis], X)
        
	if normalized_y:
		Y = np.multiply((1.0 / np.linalg.norm(Y, axis=1))[:, np.newaxis], Y)
        
	return X, Y


def gen_feat_label_dicts(train_feats_labels_list, test_feats_labels_list):
	feat_set = set([])
	label_set= set([])

	for fll in train_feats_labels_list:
		feat_set.update(set(fll[0].keys()))
		label_set.update((set(fll[1])))
	for fll in test_feats_labels_list:
		feat_set.update(set(fll[0].keys()))
		label_set.update((set(fll[1])))
    
	feat_dic = {k: v for v, k in enumerate(feat_set)}
	label_dic = {k: v for v, k in enumerate(label_set)}
	return feat_dic, label_dic


def gen_data_cluster(feats_labels_list, feat_dic, label_dic, normalized_x=True, normalized_y=True):

	num_samples = len(feats_labels_list)
	X = np.zeros((num_samples, len(feat_dic)), dtype='float32')
	Y = np.zeros((num_samples, len(label_dic)), dtype='float32')
	
	for i, (feats, labels) in enumerate(feats_labels_list[:]):
		#print(feats.items())
		X[i, [feat_dic[item] for item in feats.keys()]] += list(feats.values())
		Y[i, [label_dic[item] for item in labels]] += 1
        
	if normalized_x:
		X = np.multiply((1.0 / np.linalg.norm(X, axis=1))[:, np.newaxis], X)
        
	if normalized_y:
		Y = np.multiply((1.0 / np.linalg.norm(Y, axis=1))[:, np.newaxis], Y)
        
	return X, Y


# select data as train split and test split
def split_data(X, Y, test_size=0.3):
	return train_test_split(pd.DataFrame(X), pd.DataFrame(Y), test_size=test_size)


# frobenius norm
def f_norm(X):
	return np.linalg.norm(X)


# l1 norm (sum of absolute values)
def l1_norm(X):
	return np.sum(np.abs(X.flatten()))


# save data and SVP results
def save_temp(temp_filename, U, Lmd, X_train, Y_train, X_dev, Y_dev, X_test, Y_test):
	np.savez_compressed(temp_filename, U=U, Lmd=Lmd, 
					X_train=X_train, Y_train=Y_train, 
					X_dev=X_dev, Y_dev=Y_dev,
					X_test=X_test, Y_test=Y_test)

    
def save_temp_index(temp_filename, U, Lmd, train_index, dev_index, test_index):
	np.savez_compressed(temp_filename, U=U, Lmd=Lmd, 
					train_index=np.array(train_index),
					dev_index=np.array(dev_index),
					test_index=np.array(test_index))


# load temp data
def load_temp(temp_filename):
	loaded = np.load(temp_filename + '.npz')
	U = loaded['U']
	Lmd = loaded['Lmd']
	X_train = loaded['X_train']
	Y_train = loaded['Y_train']
	X_dev = loaded['X_dev']
	Y_dev = loaded['Y_dev']
	X_test = loaded['X_test']
	Y_test = loaded['Y_test']
	return U, Lmd, X_train, Y_train, X_dev, Y_dev, X_test, Y_test


def load_temp_index(temp_filename):
	loaded = np.load(temp_filename + '.npz')
	U = loaded['U']
	Lmd = loaded['Lmd']
	train_index = loaded['train_index']
	dev_index = loaded['dev_index']
	test_index = loaded['test_index']
	return U, Lmd, train_index, dev_index, test_index