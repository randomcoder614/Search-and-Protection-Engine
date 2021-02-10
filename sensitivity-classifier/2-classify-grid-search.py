import os.path
import datetime

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.metrics import f1_score, fbeta_score, balanced_accuracy_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from pprint import pprint
from time import time
import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
import re



train_file = 'not-judged-docs.txt'
test_file = 'judged-docs.txt'


porter_stemmer = PorterStemmer()

# Use NLTK's PorterStemmer
def stemming_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [porter_stemmer.stem(word) for word in words]
    #words = [wordnet_lemmatizer.lemmatize(word) for word in words]
    return words



def parse_samples_file(filename):
    print datetime.datetime.now(), 'Parsing file', filename
    features = []
    labels = []
    id_to_text = {}
    id_to_label = {}
    with open(filename) as f:
        line = f.readline() # skip the header line
        line = f.readline()
        while line:
            tokens = line.split('\t')
            features.append(tokens[4].lower() + ' ' + tokens[5].lower())
            labels.append(int(tokens[2]))
            id_to_text[int(tokens[0])] = features[-1]
            id_to_label[int(tokens[0])] = labels[-1]
            line = f.readline()
    return features, labels, id_to_text, id_to_label

def cross_validate_best_idf_stemming_c_threshold(train_paragraphs, train_labels):
    # Do cross validation on training data to get the best threshold
    use_idfs = [False, True]
    stemmers = [None, stemming_tokenizer]
    cs = [0.01, 0.1, 1, 5, 10]
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    f1_scores = {}
    for use_idf in use_idfs:
        f1_scores[use_idf] = {}
        for stemmer in stemmers:
            f1_scores[use_idf][stemmer] = {}
            for c in cs:
                f1_scores[use_idf][stemmer][c] = {}
                for threshold in thresholds:
                    f1_scores[use_idf][stemmer][c][threshold] = 0.0
    best_use_idf = -1
    best_stemmer = -1
    best_c = -1
    best_threshold = -1
    # I'm optimizing towards the best f1 score.
    best_f1_score = 0.0
    for use_idf in use_idfs:
        for stemmer in stemmers:
            for c in cs:
                print 'Combination use_idf =', use_idf, 'stemmer =', stemmer, 'c = ', c
                kfold = KFold(5, True, random_state=1)
                for train_indicies, val_indicies in kfold.split(train_paragraphs):
                    t_sentences = [train_paragraphs[i] for i in train_indicies]
                    v_sentences = [train_paragraphs[i] for i in val_indicies]
                    t_labels = [train_labels[i] for i in train_indicies]
                    v_labels = [train_labels[i] for i in val_indicies]

                    vect = TfidfVectorizer(stop_words='english', use_idf=use_idf, tokenizer=stemmer)
                    clf = LogisticRegression(class_weight='balanced', C=c)
                    pipeline = Pipeline([
                        ('vect',vect),
                        ('clf',clf)
                    ])
                    pipeline.fit(t_sentences, t_labels)
                    probs = pipeline.predict_proba(v_sentences)
                    for index, t in enumerate(thresholds):
                        temp_preds = []
                        for prob in probs:
                            if prob[1] >= t:
                                temp_preds.append(1)
                            else:
                                temp_preds.append(0)
                        f1_scores[use_idf][stemmer][c][t] = f1_score(v_labels, temp_preds)
    print f1_scores
    for use_idf in use_idfs:
        for stemmer in stemmers:
            for c in cs:
                for threshold in thresholds:
                    if(f1_scores[use_idf][stemmer][c][threshold] > best_f1_score):
                        best_threshold = threshold
                        best_c = c
                        best_use_idf = use_idf
                        best_stemmer = stemmer
                        best_f1_score = f1_scores[use_idf][stemmer][c][threshold]
                        print best_f1_score, 'when use_idf=', best_use_idf, 'stemmer=', best_stemmer, 'c=', best_c, 'threshold=', best_threshold
    return best_use_idf, best_stemmer, best_c, best_threshold




## Main code to run
train_x, train_y, train_id_to_text, train_id_to_label = parse_samples_file(train_file)
test_x, test_y, test_id_to_text, test_id_to_label = parse_samples_file(test_file)

print len(train_x), len(train_y), len(test_x), len(test_y)

#best_use_idf, best_stemmer, best_c, best_threshold = cross_validate_best_idf_stemming_c_threshold(train_x, train_y)
#print '--------------- Best use_idf, stemmer, c, threshold', best_use_idf, best_stemmer, best_c, best_threshold

#Best use_idf, stemmer, c, threshold True <function stemming_tokenizer at 0x7fe6d6626ed0> 5 0.8
best_use_idf = True
best_stemmer = stemming_tokenizer
best_c = 5
best_threshold = 0.8


vect = TfidfVectorizer(stop_words='english', use_idf=best_use_idf, tokenizer=best_stemmer)
clf = LogisticRegression(class_weight='balanced', C=best_c)
pipeline = Pipeline([
    ('vect',vect),
    ('clf',clf)
])
pipeline.fit(train_x, train_y)

y_probs = pipeline.predict_proba(test_x)
y_preds = []
for y in y_probs:
    if y[1] >= best_threshold:
        y_preds.append(1)
    else:
        y_preds.append(0)
print confusion_matrix(test_y, y_preds)
print(classification_report(test_y, y_preds, target_names=['0', '1']))
print 'P', precision_score(test_y, y_preds)
print 'R', recall_score(test_y, y_preds)
print 'ACC', accuracy_score(test_y, y_preds)
print 'F2', fbeta_score(test_y, y_preds, beta=2)
print 'BAC', balanced_accuracy_score(test_y, y_preds)


index = 0
f = open('ohsumed-labels-probabilities.csv', 'w')
f.write('seq,id,probability,1-probability,prediction,label\n')
for docid in test_id_to_text:
    y_prob = pipeline.predict_proba([test_id_to_text[docid]])
    pred = 0
    if y_prob[0][1] >= best_threshold:
        pred = 1
    f.write(str(index) + ',' + str(docid) + ',' + str(y_prob[0][1]) + ',' + str(y_prob[0][0]) + ',' + str(pred) + ',' + str(test_id_to_label[docid]) + '\n')
f.close()




