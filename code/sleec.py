import evaluation as ev

import ast
import re
import numpy as np
import copy

from scipy.spatial import distance
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_similarity_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


# check math kernel
def check_mkl():
	np.show_config()


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
	
	num_ents = len(feats_labels_list)
	X = np.zeros((spec_tup[0], spec_tup[1]), dtype='float32')
	Y = np.zeros((spec_tup[0], spec_tup[2]), dtype='float32')
	
	for i, (feats, labels) in enumerate(feats_labels_list[:]):
		#print(feats.items())
		X[i, list(feats.keys())] += list(feats.values())
		Y[i, labels] += 1
        
	if normalized_x:
		X = np.multiply((1.0 / np.linalg.norm(X, axis=1))[:, np.newaxis], X)
        
	if normalized_y:
		Y = np.multiply((1.0 / np.linalg.norm(Y, axis=1))[:, np.newaxis], Y)
        
	return X, Y


# select data as train split and test split
def split_data(X, Y, test_size=0.3):
	return train_test_split(X, Y, test_size=test_size)


# distance metric for length-normalized vectors
def mydist(u, v):
	return 1.0 - np.dot(u, v)


# return the distances and indices of k nearest neighbors
def knn(Y_train, metric, alpha=0.2):
	n_neighbors = int(alpha * len(Y_train))
	print('k:', n_neighbors)
	neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric=metric)
	neigh.fit(Y_train)
	distances, indices = neigh.kneighbors(Y_train)
	return distances, indices


# preserve only elements at indices, other masked to zeros
def P_Omega(X, indices):
	X_ret = np.zeros(X.shape, dtype='complex128')
	for i, row in enumerate(indices):
		X_ret[i, row] = X[i, row]
	return X_ret


# frobenius norm
def f_norm(X):
	return np.linalg.norm(X)


# l1 norm (sum of absolute values)
def l1_norm(X):
	return np.sum(np.abs(X.flatten()))


# SVP
def SVP(Y_train, emb_size, indices, eta=0.2, max_iteration=200):
	
	def top_eigen_decomp(X):
		w, v = np.linalg.eigh(X)
		sorded_idx = np.argsort(w)[:: -1][: emb_size]
		U_M = v[:, sorded_idx]
		Lmd_M = np.diag(w[sorded_idx])
		Lmd_M[Lmd_M < 0] = 0
		return U_M, Lmd_M
	
	# initalization
	G = np.matmul(Y_train, Y_train.T)
	M = np.zeros(G.shape)
	P_G = P_Omega(G, indices)
	P_M = P_Omega(M, indices)
	U_op = None
	Lmd_op = None
	loss_op = np.inf
	
	for i in range(max_iteration):
		M_hat = M + eta * (G - P_M)
		U, Lmd = top_eigen_decomp(M_hat)
		M = np.matmul(np.matmul(U, Lmd), U.T)
		P_M = P_Omega(M, indices)
		loss = f_norm(P_G - P_M)
		
		if loss < loss_op:
			U_op = U
			Lmd_op = Lmd
			loss_op = loss
			print('iter:', i, '; loss:', loss)
		else: 
			print('Convergence.')
			return U_op, Lmd_op
	
	print('Max_iteration reached.')
	return U, Lmd


# save data and SVP results
def save_temp(temp_filename, U, Lmd, X_train, Y_train, X_dev, Y_dev, X_test, Y_test):
	np.savez_compressed(temp_filename, U=U, Lmd=Lmd, 
					X_train=X_train, Y_train=Y_train, 
					X_dev=X_dev, Y_dev=Y_dev,
					X_test=X_test, Y_test=Y_test)


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


# initialize Z
def get_init_Z(U, Lmd):
	return np.real(np.matmul(U, np.sqrt(Lmd)))


# ADMM
def ADMM(X, Z, rho=0.1, lmd=0.4, mu=0.01, max_iteration=200, verbose=False):
	
	def mask_alpha(arr):
		func = lambda x: np.abs(x) - mu / rho
		ret = func(arr)
		ret[ret < 0] = 0
		ret = np.multiply(np.sign(arr), ret)
		return ret
	
	def gen_default_params():
		m, n = X.shape
		w, v = np.linalg.eig(X.T.dot(X))
		rho = 1 / np.amax(np.absolute(w))
		lmd = np.sqrt(2 * np.log10(n))
		
		print(rho, lmd)
		return rho, lmd
	
	alpha = np.zeros(Z.shape, dtype='float32')
	beta = np.zeros(Z.shape, dtype='float32')
	print(X.shape)
	G = np.matmul(X.T, X)
	I_size = G.shape[0]
	V_op = None
	loss_op = np.inf
	
	for i in range(max_iteration):
		
		Q = np.matmul(X.T, (Z + rho * (alpha - beta)))
		V = np.matmul(np.mat(G * (1 + rho) + lmd * np.identity(I_size)).I, Q)
		alpha = np.matmul(X, V) + beta        
		alpha = mask_alpha(alpha)
		beta = beta + np.matmul(X, V) - alpha
		loss = f_norm(Z - np.matmul(X, V)) ** 2 + lmd * f_norm(V) ** 2 + mu * l1_norm(np.matmul(X, V))
		
		if loss < loss_op:
			loss_op = loss
			V_op = V
			if verbose:
				print('iter:', i, '; loss:', loss)
		else:
			if verbose:
				print('iter:', i, '; loss:', loss)
				print('Convergence.')
			return V_op
	
	if verbose:
		print('Max_iteration reached.')
	
	return V


# test algorithm
def test_alg(X_test, Y_train, Z_final, V, top_p):
	#   Z_final: final embeddings of labels
	#         V: regressor learned by ADMM
	#     top_p: number of neighbors to add
	#         Z: predicted embeddings of X_test
	#     L_hat: predicted labels
	# distances: distances to nearest top_p neighbors

	Z = np.matmul(X_test, V)
	L_hat = []
	distances = []

	cdist = distance.cdist(Z, Z_final, 'cosine')
	idx = cdist.argsort(axis=1)[:, : top_p]

	for i in range(len(X_test)):
		L_hat.append(np.sum(np.array(Y_train)[idx[i]], axis=0))

	return np.array(L_hat), np.array(distances)


# experiment with different parameters for ADMM and nearest neighbors number used to predict labels
def experiment(X_train, Y_train, X_test, Y_test, X_test2, Y_test2, Z, rho=0.05, gen_curve=True,
			   lmd=np.arange(0.1, 1.1, 0.1), 
			   mu=np.arange(0.01, 0.06, 0.01), 
			   nn=[5, 10, 15, 20, 25], 
			   k_list=[1, 2, 3, 4, 5]):
	
	combs = np.array(np.meshgrid(lmd, mu)).T.reshape(-1,2)
	specs = []
	Y_test_hot = copy.deepcopy(Y_test)
	Y_test_hot[Y_test_hot > 0] = 1
	idx_to_find = ev.find_top_labels(Y_test_hot)
	Y_test2_hot = copy.deepcopy(Y_test2)
	Y_test2_hot[Y_test2_hot > 0] = 1
	idx2_to_find = ev.find_top_labels(Y_test2_hot)
    
	print('=' * 50)
	
	for tup in combs: 
		print()
		print('Specs: lmd={0[0]}, mu={0[1]}'.format(tup))
		V_test = ADMM(X_train, Z, rho=rho, lmd=tup[0], mu=tup[1])
		Z_final = np.matmul(X_train, V_test)

		for n in nn:
			print('#### nn:', n)
            # for dev
			L_hat, dists = test_alg(X_test[:], Y_train, Z_final, V_test, n)
			L_hat_hot = copy.deepcopy(L_hat)
			L_hat_hot[L_hat_hot > 0] = 1
			eval_ret = ev.eval_all(L_hat, Y_test, L_hat_hot, Y_test_hot, idx_to_find, k_list, gen_curve=gen_curve) 
            
			# for test
			L_hat2, dists2 = test_alg(X_test2[:], Y_train, Z_final, V_test, n)
			L_hat2_hot = copy.deepcopy(L_hat2)
			L_hat2_hot[L_hat2_hot > 0] = 1
			eval_ret2 = ev.eval_all(L_hat2, Y_test2, L_hat2_hot, Y_test2_hot, idx2_to_find, k_list, gen_curve=gen_curve) 

			specs.append(((tup[0], tup[1], n), eval_ret, eval_ret2))
		print()
		print('=' * 50)
	
	return specs