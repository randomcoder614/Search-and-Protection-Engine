import os.path
import datetime
import os
import math
import sys
from collections import defaultdict
import csv
from scipy import stats


# To get the ranking you want to evaluate
ranking_file = sys.argv[1]
# To get the true relevance labels, e.g. ohsumed/Fold1/
qrel_folder = sys.argv[2]
# To get the true sensitivity labels and probabilities, e.g. ohsumed/ohsumed-labels-probabilities.csv
sensitivity_file = sys.argv[3]

# e.g. ohsumed-sense/output1/indri-10allk1.txt
baseline_ranking_file = sys.argv[4]

KST_CONSTANT = 40
KST_BALANCE_CONSTANT = 1.5

TOP_K = 10

#SENSIVITY_PENALTIES = [3, 6, 15, 100, 1000, 10000]
SENSIVITY_PENALTIES = [0, 1, 3]

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
ranked_list = parse_ranking_file(ranking_file, qrel)

#0) Apply filter on the baseline and save scores
baseline_ranked_list = parse_ranking_file(baseline_ranking_file, qrel)
baseline_scores = []
for i in range(len(SENSIVITY_PENALTIES)):
	baseline_scores.append([0]*(len(baseline_ranked_list)))
for qid in range(1, len(baseline_ranked_list)+1):
	sorted_docs = sorted(baseline_ranked_list[qid], key=lambda doc: doc[2] if pred_sensitivity_label[doc[0]] == 0 else -float('inf'), reverse=True) # sort by new_score if not sensitive
	doc_array = qrel[qid].keys()
	for index, penalty in enumerate(SENSIVITY_PENALTIES):
		sense_score = compute_sense_measure(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, TOP_K, penalty)
		#sense_score = compute_alpha_ncs_dcg(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, TOP_K, 12, 1)
		if penalty == 1:
			print('baseline', qid, sense_score)
		baseline_scores[index][qid-1] = sense_score
for i in range(len(SENSIVITY_PENALTIES)):
	print(round(sum(baseline_scores[i])/len(baseline_scores[i]),3), end=' ')
print()


# 1) Get results for no filter approach
#print '------------ NO FILTER'
averages = [0]*len(SENSIVITY_PENALTIES)
scores = []
for i in range(len(SENSIVITY_PENALTIES)):
	scores.append([0]*(len(ranked_list)))
for qid in range(1, len(ranked_list)+1):
	sorted_docs = sorted(ranked_list[qid], key=lambda doc: doc[2], reverse=True) # sort by new_score
	#print qid, sorted_docs
	doc_array = qrel[qid].keys()
	ndcg_score = compute_ndcg(sorted_docs, doc_array, qrel[qid], TOP_K)
	for index, penalty in enumerate(SENSIVITY_PENALTIES):
		#print 'Penalty =', penalty
		#sense_score = compute_sense_measure(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, TOP_K, penalty)
		sense_score = compute_alpha_ncs_dcg(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, TOP_K, 12, 1)
		scores[index][qid-1] = sense_score
		for i in range(TOP_K):
			docid = sorted_docs[i][0]
		#print qid, sense_score, '\t\t',
		averages[index] += sense_score
	#print
averages = [a/len(ranked_list) for a in averages]
for index, a in enumerate(averages):
	# Denominator is the same here, so we compare the sum only
	print(str(round(a, 3)) + ('*' if sum(scores[index]) > sum(baseline_scores[index]) and stats.ttest_rel(scores[index], baseline_scores[index])[1] < 0.05 else ''), end=' & ')
print()


# 2) Get results for pre/post filter approach
#print '------------ PRE/POST FILTER'
averages_filter = [0]*len(SENSIVITY_PENALTIES)
scores_filter = []
for i in range(len(SENSIVITY_PENALTIES)):
	scores_filter.append([0]*(len(ranked_list)))
for qid in range(1, len(ranked_list)+1):
	sorted_docs = sorted(ranked_list[qid], key=lambda doc: doc[2] if pred_sensitivity_label[doc[0]] == 0 else -float('inf'), reverse=True) # sort by new_score if not sensitive
	#print qid, sorted_docs
	doc_array = qrel[qid].keys()
	ndcg_score = compute_ndcg(sorted_docs, doc_array, qrel[qid], TOP_K)
	for index, penalty in enumerate(SENSIVITY_PENALTIES):
		#print 'Penalty =', penalty
		sense_score = compute_sense_measure(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, TOP_K, penalty)
		#sense_score = compute_alpha_ncs_dcg(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, TOP_K, 12, 1)
		if penalty == 1:
			print('approach', qid, sense_score)
		scores_filter[index][qid-1] = sense_score
		for i in range(TOP_K):
			docid = sorted_docs[i][0]
		#print qid, sense_score, '\t\t',
		averages_filter[index] += sense_score
	#print
averages_filter = [a/len(ranked_list) for a in averages_filter]
for index, a in enumerate(averages_filter):
	# Denominator is the same here, so we compare the sum only
	print(str(round(a, 3)) + ('*' if sum(scores_filter[index]) > sum(baseline_scores[index]) and stats.ttest_rel(scores_filter[index], baseline_scores[index])[1] < 0.05 else ''), end=' & ')
print()


def optimize_linear_combination(ranked_list, validation_qids, penalty, topK):
	possible_values = [2, 4, 8, 16]
	max_score = (-1)*penalty*len(validation_qids) - 1
	#print(max_score)
	max_p = -1
	for p in possible_values:
		current_score = 0
		for qid in validation_qids:
			minscore = min([s[2] for s in ranked_list[qid]])
			maxscore = max([s[2] for s in ranked_list[qid]])
			sorted_docs = sorted(ranked_list[qid], key=lambda doc: normalize_score(doc[2], minscore, maxscore) + p*(1 - prob_sensitivity[doc[0]]), reverse=True) # sort by new_score if not sensitive
			doc_array = qrel[qid].keys()
			current_score += compute_sense_measure(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, TOP_K, penalty)
		if current_score > max_score:
			max_score = current_score
			max_p = p
	return max_p


# 3) Get results for linear combination approach (rel_score + 2*(1-probability of sensitivity))
#print '------------ LINEAR COMBINATION'
validation_qids = [
[85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106],
[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
[22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42],
[43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63],
[64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84],
[64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84] # repeat this row again just to handle when qid = 106
]
averages_linear = [0]*len(SENSIVITY_PENALTIES)
scores_linear = []
for i in range(len(SENSIVITY_PENALTIES)):
	scores_linear.append([0]*(len(ranked_list)))
for qid in range(1, len(ranked_list)+1):
	minscore = min([s[2] for s in ranked_list[qid]])
	maxscore = max([s[2] for s in ranked_list[qid]])
	for index, penalty in enumerate(SENSIVITY_PENALTIES):
		max_p = optimize_linear_combination(ranked_list, validation_qids[int((qid-1)/21)], penalty, 10)
		#print('----------------max_p', max_p)
		sorted_docs = sorted(ranked_list[qid], key=lambda doc: normalize_score(doc[2], minscore, maxscore) + 10*(1 - prob_sensitivity[doc[0]]), reverse=True) # sort by new_score if not sensitive
		doc_array = qrel[qid].keys()
		#sense_score = compute_sense_measure(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, TOP_K, penalty)
		sense_score = compute_alpha_ncs_dcg(sorted_docs, doc_array, qrel[qid], true_sensitivity_label, TOP_K, 12, 1)
		scores_linear[index][qid-1] = sense_score
		for i in range(TOP_K):
			docid = sorted_docs[i][0]
		averages_linear[index] += sense_score
averages_linear = [a/len(ranked_list) for a in averages_linear]
for index, a in enumerate(averages_linear):
	# Denominator is the same here, so we compare the sum only
	print(str(round(a, 3)) + ('*' if sum(scores_linear[index]) > sum(baseline_scores[index]) and stats.ttest_rel(scores_linear[index], baseline_scores[index])[1] < 0.05 else ''), end=' & ')
print()
