import os.path
import datetime
import os
import math
import sys
from collections import defaultdict
import csv
from scipy import stats
import collections
import statistics

# To get the ranking you want to evaluate
ranking_file = sys.argv[1]
# To get the true relevance labels 
qrel_folder = sys.argv[2]
# To get the true sensitivity labels and probabilities, e.g. john/john-labels-probabilities.csv
sensitivity_file = sys.argv[3]

penalty = int(sys.argv[4])

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
			print('Error: can\'t find the best document'), 
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
			print('Error: can\'t find the best document'), 
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
ranked_list = parse_ranking_file(ranking_file, qrel)

def optimize_cutoff(ranked_list, validation_qids, penalty):
	#possible_values = [0,1,2,3,4,5,6,7,8,9,10]
	possible_values = [0,1,2,3,4,5]
	max_score = (-1)*penalty*len(validation_qids) - 1
	max_cutoff = -1
	for c in possible_values:
		current_score = 0
		for qid in validation_qids:
			sorted_docs = sorted(ranked_list[qid], key=lambda doc: doc[2], reverse=True) # sort by new_score if not sensitive
			doc_array = qrel[qid].keys()
			current_score += compute_sense_measure(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, c, penalty)
		if current_score > max_score:
			max_score = current_score
			max_cutoff = c
	return max_cutoff


validation_qids = [
[85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106],
[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
[22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42],
[43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63],
[64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84],
[64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84] # repeat this row again just to handle when qid = 106
]


orig_scores = {}
static_scores = {}
dynamic_scores = {}


# 1) Get results for no filter approach
for tk in range(0,11):
	avg_score = 0
	for qid in range(1, len(ranked_list)+1):
		sorted_docs = sorted(ranked_list[qid], key=lambda doc: doc[2], reverse=True) # sort by new_score
		doc_array = qrel[qid].keys()
		avg_score += compute_sense_measure(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, tk, penalty)
		orig_scores[qid] = compute_sense_measure(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, 10, penalty)
	avg_score /= len(ranked_list)
	#print('TOP_K =', tk, 'avg. sense score =', avg_score)
	#print(tk,round(avg_score,3))
#print()

# 2) Static cutoff
avg_score = 0
for qid in range(1, len(ranked_list)+1):
	sorted_docs = sorted(ranked_list[qid], key=lambda doc: doc[2], reverse=True) # sort by new_score
	doc_array = qrel[qid].keys()
	best_cutoff = optimize_cutoff(ranked_list, validation_qids[int((qid-1)/21)], penalty)
	#print('----------',best_cutoff)
	avg_score += compute_sense_measure(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, best_cutoff, penalty)
	static_scores[qid] = compute_sense_measure(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, best_cutoff, penalty)
avg_score /= len(ranked_list)
#print(round(avg_score,3))


def compute_sensitive_percentage(doc_array, sensitivity_dict):
	count = 0
	for d in doc_array:
		if sensitivity_dict[d] != 0:
			count += 1
	return count*1.0/len(doc_array)

def optimize_starting_rank(ranked_list, validation_qids, penalty):
	possible_values = [3, 5, 7, 10]
	max_score = (-1)*penalty*len(validation_qids) - 1
	max_starting_rank = -1
	for s in possible_values:
		current_score = 0
		for qid in validation_qids:
			sorted_docs = sorted(ranked_list[qid], key=lambda doc: doc[2], reverse=True) # sort by new_score if not sensitive
			doc_array = qrel[qid].keys()
			sensitive_percent = compute_sensitive_percentage(doc_array, pred_sensitivity_label)
			cutoff = int(s*(1-sensitive_percent))
			current_score += compute_sense_measure(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, cutoff, penalty)
		if current_score > max_score:
			max_score = current_score
			max_starting_rank = s
	return max_starting_rank


# 3) Dynamic cutoff per query
avg_score = 0
for qid in range(1, len(ranked_list)+1):
	sorted_docs = sorted(ranked_list[qid], key=lambda doc: doc[2], reverse=True) # sort by new_score
	doc_array = qrel[qid].keys()
	max_starting_rank = optimize_starting_rank(ranked_list, validation_qids[int((qid-1)/21)], penalty)
	max_starting_rank = 6
	sensitive_percent = compute_sensitive_percentage(doc_array, pred_sensitivity_label)
	best_cutoff = int(max_starting_rank * (1-sensitive_percent))
	#print(max_starting_rank, '----------',best_cutoff)
	avg_score += compute_sense_measure(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, best_cutoff, penalty)
	dynamic_scores[qid] = compute_sense_measure(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, best_cutoff, penalty)
avg_score /= len(ranked_list)
#print(round(avg_score,3))


# Printing average score for orginal@10, static cutoff, and dynamic cutoff
sortedKeys = sorted(orig_scores)
orig_values = []
static_values = []
dynamic_values = []
for k in sortedKeys:
	orig_values.append(orig_scores[k])
	static_values.append(static_scores[k])
	dynamic_values.append(dynamic_scores[k])
print(round(statistics.mean(orig_values),3))
print(str(round(statistics.mean(static_values),3)) + ('*' if (stats.kendalltau(orig_values, static_values)[1] < 0.05) else ''))
print(str(round(statistics.mean(dynamic_values),3)) + ('*' if (stats.kendalltau(orig_values, dynamic_values)[1] < 0.05) else ''))

'''
# Printing score per query and sorted
orig_scores = sorted(orig_scores.items(), key=lambda kv: kv[1])
orig_scores = collections.OrderedDict(orig_scores)
#print(orig_scores)
for index, qid in enumerate(orig_scores.keys()):
	print(index+1, qid, round(orig_scores[qid],3), round(static_scores[qid],3), round(dynamic_scores[qid],3))
'''
