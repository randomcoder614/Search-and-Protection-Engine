import os.path
import datetime
import os
import math
import sys
from collections import defaultdict
import csv


from scipy.stats import kendalltau


# To get the ranking you want to evaluate
ranking_files = ['ohsumed-sense/output1/indri-4allk1.txt', 'ohsumed-sense/output4/indri-ncs_dcg-4allk12.txt', 'ohsumed-sense/output4/indri-4allk0.txt', 'ohsumed-sense/output4/indri-4allk1.txt', 'ohsumed-sense/output4/indri-4allk3.txt',
'ohsumed-sense-bert/output1/indri-4allk1.txt', 'ohsumed-sense-bert/output4/indri-4allk0.txt', 'ohsumed-sense-bert/output4/indri-4allk1.txt', 'ohsumed-sense-bert/output4/indri-4allk3.txt']
ranking_files_bert_t5 = ['../test-pygaggle/result-BERT-ohsumed.out', '../test-pygaggle/result-T5-ohsumed.out']


# To get the true relevance labels 
qrel_folder = 'OHSUMED/QueryLevelNorm/Fold1/'

# To get the true sensitivity labels and probabilities, e.g. ohsumed/ohsumed-labels-probabilities.csv
sensitivity_file = 'ohsumed-labels-probabilities.csv'


KST_CONSTANT = 40
KST_BALANCE_CONSTANT = 1.5

TOP_K = 10

#SENSIVITY_PENALTIES = [3, 6, 15, 100, 1000, 10000]
#SENSIVITY_PENALTIES = [3, 15, 10000]
SENSIVITY_PENALTIES = [0, 1, 3]
#ALPHA = 0.00001


# Parses sensitivity labels file and result is dict, key = docid (string) and value = sensitivity (0 or 1 int)
def parse_true_sensivity_labels_file(filename):
	with open(filename, mode='r') as infile:
		reader = csv.reader(infile)
		# skip header
		next(reader)
		mydict = dict((rows[1], int(rows[-1])) for rows in reader)
		return mydict

# Parses sensitivity labels file and result is dict, key = docid (string) and value = sensitivity (0 or 1 int)
def parse_pred_sensivity_labels_file(filename):
	with open(filename, mode='r') as infile:
		reader = csv.reader(infile)
		# skip header
		next(reader)
		mydict = dict((rows[1], int(rows[-2])) for rows in reader)
		return mydict

# Parses sensitivity labels file and result is dict, key = docid (string) and value = prob of sensitivity [0, 1]
def parse_prob_sensivity_labels_file(filename):
	with open(filename, mode='r') as infile:
		reader = csv.reader(infile)
		# skip header
		next(reader)
		mydict = dict((rows[1], float(rows[2])) for rows in reader)
		return mydict

# Parses qrel file and result is dict, key = qid (int) and value = dict of doc_id (string) and rel_scores(int)
def parse_qrel_folder(foldername):
	qrel = {}
	for filename in os.listdir(qrel_folder):
		f = open(qrel_folder+filename)
		for line in f:
			tokens = line.split(' ')
			qid = int(tokens[1].split(':')[1])
			if not qid in qrel:
				qrel[qid] = {}
			rel_score = int(tokens[0])
			docid = tokens[-1].strip()
			qrel[qid][docid] = rel_score
	return qrel

# Parses ranking file and result is dict, key = qid (int) and value = array of tuples (email_id(string), orig_score(int), new_score(float))
def parse_ranking_file(filename, qrel):
	ranked_list = {}
	current_qid = -1
	f = open(filename, 'r')
	for line in f:
		tokens = line.split(' ')
		qid = int(tokens[0])
		if not qid in ranked_list:
			ranked_list[qid] = []
		current_qid = qid
		email_id = tokens[4]
		orig_score = int(qrel[qid][email_id]) # Just get the original score from the parsed qrel
		new_score = float(tokens[6])
		ranked_list[current_qid].append((email_id, orig_score, new_score))
	return ranked_list


# Parses ranking file and result is dict, key = qid (int) and value = array of tuples (email_id(string), orig_score(int), new_score(float))
def parse_ranking_file_BERT_T5(filename):
	ranked_list = {}
	current_qid = -1
	f = open(filename, 'r')
	for line in f:
		tokens = line.split(' ')
		if tokens[0].strip() == 'Query':
			qid = int(tokens[1])
			if not qid in ranked_list:
				ranked_list[qid] = []
			current_qid = qid
		elif len(tokens) >= 8 and tokens[3].strip() == 'original':
			email_id = tokens[0].strip()
			orig_score = int(tokens[5])
			new_score = float(tokens[8])
			ranked_list[current_qid].append((email_id, orig_score, new_score))
	return ranked_list


def gain(rel_value):
	return 2.0**rel_value - 1

def discount(rank_value):
	return 1.0/float(math.log(rank_value + 1, 2))

# Returns the standard CS_DCG as in SIGIR '19 paper
def compute_standard_cs_dcg(ranking, doc_array, doc_rel, sensitivity_dict, top_k, SENSIVITY_PENALTY):
	result = 0
	for i in range(top_k):
		docid = ranking[i][0]
		result += gain(doc_rel[docid]) * discount(i+1) - SENSIVITY_PENALTY * sensitivity_dict[docid]
	return result

# Returns the alpha CS_DCG as in SIGIR '19 paper
def compute_alpha_cs_dcg(ranking, doc_array, doc_rel, sensitivity_dict, top_k, SENSIVITY_PENALTY, ALPHA):
	result = 0
	count_sensitive_docs = 0
	for i in range(top_k):
		docid = ranking[i][0]
		result += gain(doc_rel[docid]) * discount(i+1) - SENSIVITY_PENALTY * sensitivity_dict[docid] * (ALPHA**count_sensitive_docs)
		if sensitivity_dict[docid] != 0:
			count_sensitive_docs += 1
	return result

def compute_best_alpha_cs_dcg(ranking, doc_array, doc_rel, sensitivity_dict, top_k, SENSIVITY_PENALTY, ALPHA):
	result = 0.0
	count_sensitive_docs = 0
	visited = {}
	for i in range(top_k):
		best_score = -float('inf')
		best_docid = 0
		for d in doc_rel:
			if not d in visited:
				score = gain(doc_rel[d]) * discount(i+1) - SENSIVITY_PENALTY * sensitivity_dict[d] * (ALPHA**count_sensitive_docs)
				if score > best_score:
					best_score = score
					best_docid = d
		if best_docid == 0:
			print('Error: can\'t find the best document') 
			raise
		else:
			visited[best_docid] = True
			result += best_score
			if sensitivity_dict[best_docid] != 0:
				count_sensitive_docs += 1
	return result


def compute_worst_alpha_cs_dcg(ranking, doc_array, doc_rel, sensitivity_dict, top_k, SENSIVITY_PENALTY, ALPHA):
	result = 0.0
	count_sensitive_docs = 0
	visited = {}
	for i in range(top_k):
		best_score = float('inf')
		best_docid = 0
		for d in doc_rel:
			if not d in visited:
				score = gain(doc_rel[d]) * discount(i+1) - SENSIVITY_PENALTY * sensitivity_dict[d] * (ALPHA**count_sensitive_docs)
				if score < best_score:
					best_score = score
					best_docid = d
		if best_docid == 0:
			print('Error: can\'t find the best document')
			raise
		else:
			visited[best_docid] = True
			result += best_score
			if sensitivity_dict[best_docid] != 0:
				count_sensitive_docs += 1
	return result

def compute_alpha_ncs_dcg(ranking, doc_array, doc_rel, sensitivity_dict, top_k, SENSIVITY_PENALTY, ALPHA):
	score = compute_alpha_cs_dcg(ranking, doc_array, doc_rel, sensitivity_dict, top_k, SENSIVITY_PENALTY, ALPHA)
	best_score = compute_best_alpha_cs_dcg(ranking, doc_array, doc_rel, sensitivity_dict, top_k, SENSIVITY_PENALTY, ALPHA)
	worst_score = compute_worst_alpha_cs_dcg(ranking, doc_array, doc_rel, sensitivity_dict, top_k, SENSIVITY_PENALTY, ALPHA)
	#print(score, best_score, worst_score)
	if worst_score == best_score:
		return 0
	else:
		return (score-worst_score)/(best_score-worst_score)


def compute_dcg(ranking, doc_array, doc_rel, top_k):
	result = 0
	for i in range(top_k):
		docid = ranking[i][0]
		result += gain(doc_rel[docid]) * discount(i+1)
	return result

def compute_best_dcg(ranking, doc_array, doc_rel, top_k):
	result = 0.0
	visited = {}
	for i in range(top_k):
		best_score = -float('inf')
		best_docid = 0
		for d in doc_rel:
			if not d in visited:
				score = gain(doc_rel[d]) * discount(i+1)
				if score > best_score:
					best_score = score
					best_docid = d
		if best_docid == 0:
			print('Error: can\'t find the best document')
			return 0
		else:
			visited[best_docid] = True
			result += best_score
	return result

def compute_ndcg(ranking, doc_array, doc_rel, top_k):
	dcg = compute_dcg(ranking, doc_array, doc_rel, top_k)
	best_dcg = compute_best_dcg(ranking, doc_array, doc_rel, top_k)
	if dcg == 0.0 or best_dcg == 0.0:
		return 0.0
	else:
		return dcg*1.0/best_dcg

# Return 0 if there is any sensitive document in the top k results
def compute_cs_dcg(ranking, doc_array, doc_rel, sensitivity_dict, top_k):
	result = 0
	for i in range(top_k):
		docid = ranking[i][0]
		# Just in case
		if sensitivity_dict[docid] != 0:
			raise
		result += gain(doc_rel[docid]) * discount(i+1)
	return result

# Return 0 if there are not enough (< k) non-sensitive documents in the whole ranking
def compute_best_cs_dcg(ranking, doc_array, doc_rel, sensitivity_dict, top_k):
	result = 0.0
	visited = {}
	# Mark all sensitive documents as visited not to select them.
	for d in doc_array:
		if sensitivity_dict[d] != 0:
			visited[d] = True
		else:
			visited[d] = False

	for i in range(top_k):
		best_score = -float('inf')
		best_docid = 0
		for d in doc_rel:
			if visited[d] == False:
				score = gain(doc_rel[d]) * discount(i+1)
				if score > best_score:
					best_score = score
					best_docid = d
		if best_docid == 0:
			print('Error: can\'t find the best document')
			raise
		else:
			visited[best_docid] = True
			result += best_score
	return result


def compute_sense_measure(ranking, doc_array, doc_rel, sensitivity_dict, top_k, penalty):
	# 1) check any sensitive document, if so return -1
	for i in range(top_k):
		docid = ranking[i][0]
		if sensitivity_dict[docid] != 0:
			return -1*penalty
	# 2) compute nDCG 
	cs_dcg = compute_cs_dcg(ranking, doc_array, doc_rel, sensitivity_dict, top_k)
	best_cs_dcg = compute_best_cs_dcg(ranking, doc_array, doc_rel, sensitivity_dict, top_k)
	if cs_dcg == 0.0 or best_cs_dcg == 0.0:
		return 0
	else:
		return cs_dcg*1.0/best_cs_dcg

def normalize_score(value, minValue, maxValue):
	if minValue == maxValue:
		return 0
	else:
		return (value-minValue)/(maxValue-minValue)


true_sensitivity_label = parse_true_sensivity_labels_file(sensitivity_file)
pred_sensitivity_label = parse_pred_sensivity_labels_file(sensitivity_file)
prob_sensitivity = parse_prob_sensivity_labels_file(sensitivity_file)
qrel = parse_qrel_folder(qrel_folder)



data = []
last_index = 0

# 1) SENS measure with penalties [0, 1, 3]
for penalty in SENSIVITY_PENALTIES:
	data.append([])
	for ranking_file in ranking_files:
		ranked_list = parse_ranking_file(ranking_file, qrel)
		avg_score = 0
		for qid in range(1, len(ranked_list)+1):
			sorted_docs = sorted(ranked_list[qid], key=lambda doc: doc[2], reverse=True) # sort by new_score
			doc_array = qrel[qid].keys()
			avg_score += compute_sense_measure(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, TOP_K, penalty)
		avg_score /= len(ranked_list)
		print(round(avg_score,2), end =" ")
		data[last_index].append(avg_score)

		avg_score = 0
		for qid in range(1, len(ranked_list)+1):
			sorted_docs = sorted(ranked_list[qid], key=lambda doc: doc[2] if pred_sensitivity_label[doc[0]] == 0 else -float('inf'), reverse=True) # sort by new_score if not sensitive
			doc_array = qrel[qid].keys()
			avg_score += compute_sense_measure(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, TOP_K, penalty)
		avg_score /= len(ranked_list)
		print(round(avg_score,2), end =" ")
		data[last_index].append(avg_score)

		'''
		avg_score = 0
		for qid in range(1, len(ranked_list)+1):
			minscore = min([s[2] for s in ranked_list[qid]])
			maxscore = max([s[2] for s in ranked_list[qid]])
			sorted_docs = sorted(ranked_list[qid], key=lambda doc: normalize_score(doc[2], minscore, maxscore) + 10*(1 - prob_sensitivity[doc[0]]), reverse=True) # sort by new_score if not sensitive
			doc_array = qrel[qid].keys()
			avg_score += compute_sense_measure(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, TOP_K, penalty)
		avg_score /= len(ranked_list)
		print(round(avg_score,2), end =" ")
		data[last_index].append(avg_score)
		'''

	for ranking_file in ranking_files_bert_t5:
		ranked_list = parse_ranking_file_BERT_T5(ranking_file)
		avg_score = 0
		for qid in range(1, len(ranked_list)+1):
			sorted_docs = sorted(ranked_list[qid], key=lambda doc: doc[2], reverse=True) # sort by new_score
			doc_array = qrel[qid].keys()
			avg_score += compute_sense_measure(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, TOP_K, penalty)
		avg_score /= len(ranked_list)
		print(round(avg_score,2), end =" ")
		data[last_index].append(avg_score)

		avg_score = 0
		for qid in range(1, len(ranked_list)+1):
			sorted_docs = sorted(ranked_list[qid], key=lambda doc: doc[2] if pred_sensitivity_label[doc[0]] == 0 else -float('inf'), reverse=True) # sort by new_score if not sensitive
			doc_array = qrel[qid].keys()
			avg_score += compute_sense_measure(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, TOP_K, penalty)
		avg_score /= len(ranked_list)
		print(round(avg_score,2), end =" ")
		data[last_index].append(avg_score)

		'''
		avg_score = 0
		for qid in range(1, len(ranked_list)+1):
			minscore = min([s[2] for s in ranked_list[qid]])
			maxscore = max([s[2] for s in ranked_list[qid]])
			sorted_docs = sorted(ranked_list[qid], key=lambda doc: normalize_score(doc[2], minscore, maxscore) + 10*(1 - prob_sensitivity[doc[0]]), reverse=True) # sort by new_score if not sensitive
			doc_array = qrel[qid].keys()
			avg_score += compute_sense_measure(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, TOP_K, penalty)
		avg_score /= len(ranked_list)
		print(round(avg_score,2), end =" ")
		data[last_index].append(avg_score)		
		'''

	print()
	last_index += 1


# 2) nCS-DCG measure with penalties [15] and alphas [1, 0.5, 0.0001]
penalty = 1000
alphas = [1, 0.1]
for alpha in alphas:
	data.append([])
	for ranking_file in ranking_files:
		ranked_list = parse_ranking_file(ranking_file, qrel)
		avg_score = 0
		for qid in range(1, len(ranked_list)+1):
			sorted_docs = sorted(ranked_list[qid], key=lambda doc: doc[2], reverse=True) # sort by new_score
			doc_array = qrel[qid].keys()
			avg_score += compute_alpha_ncs_dcg(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, TOP_K, penalty, alpha)
		avg_score /= len(ranked_list)
		print(round(avg_score,2), end =" ")
		data[last_index].append(avg_score)

		avg_score = 0
		for qid in range(1, len(ranked_list)+1):
			sorted_docs = sorted(ranked_list[qid], key=lambda doc: doc[2] if pred_sensitivity_label[doc[0]] == 0 else -float('inf'), reverse=True) # sort by new_score if not sensitive
			doc_array = qrel[qid].keys()
			avg_score += compute_alpha_ncs_dcg(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, TOP_K, penalty, alpha)
		avg_score /= len(ranked_list)
		print(round(avg_score,2), end =" ")
		data[last_index].append(avg_score)

		'''
		avg_score = 0
		for qid in range(1, len(ranked_list)+1):
			minscore = min([s[2] for s in ranked_list[qid]])
			maxscore = max([s[2] for s in ranked_list[qid]])
			sorted_docs = sorted(ranked_list[qid], key=lambda doc: normalize_score(doc[2], minscore, maxscore) + 10*(1 - prob_sensitivity[doc[0]]), reverse=True) # sort by new_score if not sensitive
			doc_array = qrel[qid].keys()
			avg_score += compute_alpha_ncs_dcg(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, TOP_K, penalty, alpha)
		avg_score /= len(ranked_list)
		print(round(avg_score,2), end =" ")
		data[last_index].append(avg_score)
		'''
	for ranking_file in ranking_files_bert_t5:
		ranked_list = parse_ranking_file_BERT_T5(ranking_file)
		avg_score = 0
		for qid in range(1, len(ranked_list)+1):
			sorted_docs = sorted(ranked_list[qid], key=lambda doc: doc[2], reverse=True) # sort by new_score
			doc_array = qrel[qid].keys()
			avg_score += compute_alpha_ncs_dcg(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, TOP_K, penalty, alpha)
		avg_score /= len(ranked_list)
		print(round(avg_score,2), end =" ")
		data[last_index].append(avg_score)

		avg_score = 0
		for qid in range(1, len(ranked_list)+1):
			sorted_docs = sorted(ranked_list[qid], key=lambda doc: doc[2] if pred_sensitivity_label[doc[0]] == 0 else -float('inf'), reverse=True) # sort by new_score if not sensitive
			doc_array = qrel[qid].keys()
			avg_score += compute_alpha_ncs_dcg(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, TOP_K, penalty, alpha)
		avg_score /= len(ranked_list)
		print(round(avg_score,2), end =" ")
		data[last_index].append(avg_score)

		'''
		avg_score = 0
		for qid in range(1, len(ranked_list)+1):
			minscore = min([s[2] for s in ranked_list[qid]])
			maxscore = max([s[2] for s in ranked_list[qid]])
			sorted_docs = sorted(ranked_list[qid], key=lambda doc: normalize_score(doc[2], minscore, maxscore) + 10*(1 - prob_sensitivity[doc[0]]), reverse=True) # sort by new_score if not sensitive
			doc_array = qrel[qid].keys()
			avg_score += compute_alpha_ncs_dcg(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, TOP_K, penalty, alpha)
		avg_score /= len(ranked_list)
		print(round(avg_score,2), end =" ")
		data[last_index].append(avg_score)
		'''
	print()
	last_index += 1


#print(data)

for i in range(len(data)):
	for j in range(-2, 0):
		coef, p = kendalltau(data[i], data[j])
		print(str(round(coef,3)), end=' & ')
	print()