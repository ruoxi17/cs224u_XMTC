import numpy as np

from sklearn import metrics


def find_top_labels(true):
	label_count = np.sum(true, axis=0)
	idx = np.argsort(label_count)[:: -1]
	return idx


def eval_all(L_hat, L, preds, true, idx_to_find, k_list, gen_curve=True):

	# P@k
	def eval_prec_at_k(k):
		acc = 0
		for i in range(len(L_hat)):
			idx = L_hat[i].argsort()[::-1][: k]
			prec = np.sum(np.equal(L_hat[i][idx] > 0 ,np.equal(L[i][idx] > 0 , L_hat[i][idx] > 0))) / k
			acc += prec 
		print('P@{}:'.format(k), acc / len(L_hat))
		return acc / len(L_hat)
	
	# Hamming loss
	def eval_hamming_loss():
		loss = 0    
		for i in range(len(L_hat)):
			loss += metrics.hamming_loss(preds[i], true[i])
		print('Hamming_loss:', loss / len(L_hat))
		return loss / len(L_hat)
	
	# Jaccard similarity
	def eval_jaccard_sim():
		acc = 0    
		for i in range(len(L_hat)):
			acc += metrics.jaccard_similarity_score(preds[i], true[i])        
		print('Jaccard similarity:', acc / len(L_hat))
		return acc / len(L_hat)
	
	# Precision, recall, F1 curve
	def eval_pre_rec_f1():

		num_labs = L.shape[1]
		pre_curv = np.zeros(num_labs)
		rec_curv = np.zeros(num_labs)
		f1_curv = np.zeros(num_labs)

		start_idx = 1
		if not gen_curve:
			start_idx = num_labs
			pre_curv = np.zeros(1)
			rec_curv = np.zeros(1)
			f1_curv = np.zeros(1)
        
		for top_n in range(start_idx, num_labs + 1):
			pre, rec, f1, spt = metrics.precision_recall_fscore_support(true[:, idx_to_find[:top_n]], preds[:, idx_to_find[:top_n]])
			pre_curv[top_n - start_idx] = np.average(pre, weights=spt)
			rec_curv[top_n - start_idx] = np.average(rec, weights=spt)
			f1_curv[top_n - start_idx] = np.average(f1, weights=spt)
			print('Precision:', pre_curv[-1])
			print('Recall:', rec_curv[-1])
			print('F1:', f1_curv[-1])
		return pre_curv.round(5), rec_curv.round(5), f1_curv.round(5)
	
	ret = []
	
	for k in k_list:
		val = eval_prec_at_k(k)
		ret.append(val)
	h_loss = eval_hamming_loss()
	j_sim = eval_jaccard_sim()
	pre_c, rec_c, f1_c = eval_pre_rec_f1()
	
	print('-' * 50)
	
	return (ret, h_loss, j_sim, pre_c, rec_c, f1_c)