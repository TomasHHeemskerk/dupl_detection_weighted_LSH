# This Python file uses the following encoding: utf-8
"""
Project created on 11th of November 2022
@author: Tomas Heemskerk (Erasmus University Rotterdam studentID: 463319)

This code contains a duplicate detection algorithm consisting of LSH pre-selection followed by MSM clustering.

Note: this code is part of an individual assignment of the course FEM21037 Computer Science for Business Analytics
"""

import numpy as np
import pandas as pd
import time
import json
import sys
from itertools import chain, combinations
from difflib import SequenceMatcher
import operator
from sklearn.cluster import AgglomerativeClustering
import random

start_time = time.time()
### Hyperparameters
#includeKVP = True
th_keymatching = 0.8  # similarity treshold for matching similar keys in featuresMap (i.e. feature descriptions)
w_seqmatch = 0.6  # should sum up to 1 with w_jacsim
w_jacsim = 0.4  # should sum up to 1 with w_seqmatch
w_brand = 2  # weight of brand in LSH
w_potmid = 3  # weight of potential modelID in LSH
rpbLSH = 12  # number of rows per band in LSH
repsLSH = 2  # number of repetitions of LSH
share_LSH = 0.5  # size of SignatureMatrix relative to Binary Input Matrix
beta = 2  # to calculate Fbeta-score of LSH
th_clust = 0.21  # distance treshold for clustering
minlength_mid = 6  # minimal number of tokens in potential model ID extracted from title
### Create dataset
data_json = open('TVs-all-merged.json')  # web product description of 1624 televisions
data = json.load(data_json)  # convert to dictionary

print('Created dictionary with ' + str(
    len(data)) + ' unique model IDs. Some model IDs occur in more than one webshop product, as the dataset contains duplicates.')


### Define data cleaning functions
def replaceunit(text):
    inch = [r' inches', "'", '”', 'in', ' inch', ' inches', 'Inches', ' Inches', '-Inch', '-Inches', '-inch', '-inches',
            '"', '""']
    for item in inch:
        text = text.replace(item, 'inch ')

    hz = [r' hz', 'hz', ' HZ.', ' Hz', 'Hz.', 'Hz']
    for item in hz:
        text = text.replace(item, 'hz ')

    cdma = [r' cd/mâ²', ' cdm2', 'cdm2', 'lm', ' lm', ' cd/m²', 'cd/m²', ' cd/m2', 'nit']
    for item in cdma:
        text = text.replace(item, 'cdma ')

    lb = [r' lb', ' lbs.', ' lb.', ' pounds', 'pounds', 'lb', 'lbs.', 'lb.', 'lb']
    for item in lb:
        text = text.replace(item, 'lb ')

    watt = [r' w', 'w', ' watt', 'watt']
    for item in watt:
        text = text.replace(item, 'watt ')

    kg = [r' kg', 'kg', 'KG', ' KG', 'Kg']
    for item in kg:
        text = text.replace(item, 'kg ')

    p = [r' p', 'p', 'i/p', ' i/p', '/24p']
    for item in p:
        text = text.replace(item, 'p ')

    return text


def cleanbrand(text):
    text = text.lower()
    return text


def cleanvalue(text):
    text = text.lower()
    text = text.replace('+', '')
    text = text.replace('-', '')
    text = text.replace('without', '-')
    text = text.replace('with', '+')
    text = replaceunit(text)
    text = text.replace('and', ' ')
    text = text.replace('|', ' ')
    text = text.replace(' x ', 'x')
    text = text.replace('no', '0')
    text = text.replace('yes', '1')
    text = text.replace('false', '0')
    text = text.replace('true', '1')
    text = text.replace(',', '')
    text = text.replace('.', '')
    text = text.replace(')', '')
    text = text.replace('(', '')
    text = text.replace('/', '')
    text = text.replace('+', '')
    text = text.replace('-', '')
    text = text.replace('&#', '')

    return text


def cleanshop(text):
    text = text.lower()
    text = text.replace('.', '')
    text = text.replace(' ', '')
    return text


def cleanvaluereading(text):
    text = text.lower()
    text = text.replace('-', '')
    text = text.replace('+', '')
    text = text.replace('with', '+')
    text = text.replace('without', '-')
    text = replaceunit(text)
    text = text.replace('|', ' ')
    text = text.replace(' and ', ' ')
    text = text.replace('no', '0')
    text = text.replace(' x ', 'x')
    text = text.replace('yes', '1')
    text = text.replace('false', '0')
    text = text.replace(',', '')
    text = text.replace('true', '1')
    text = text.replace('.', '')
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('/', '')
    text = text.replace('+', '')
    text = text.replace('&#', '')
    text = text.replace('-', '')

    return text


def cleantitle(text):
    text = replaceunit(text)
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('[', '')
    text = text.replace(']', '')
    text = text.replace('.0', '')
    text = text.replace('/', '')
    text = text.replace(',', '')
    text = text.replace('inchwatt', 'inch')
    text = ' '.join(w for w in text.split() if any(x.isdigit() for x in w))

    return text

### Reading and matching the keys from the key-value pairs in the Features Map of the webshop products
def mysim(a, b):
    s1 = set(a)
    s2 = set(b)
    if min(len(s1), len(s2)) == 0:
        return 0.0
    else:
        return len(s1.intersection(s2)) / min(len(s1), len(s2))


def similar(a, b, w1, w2):
    return w1 * SequenceMatcher(None, a, b).ratio() + w2 * mysim(a.split(), b.split())


keyfeatures = {}  # dictionary with all features that are mentioned in at least two webshop products, including frequency (how often they are mentioned)
for key in data.keys():
    for i in range(len(data[key])):
        for k in data[key][i]['featuresMap'].keys():
            if k not in keyfeatures.keys():
                keyfeatures[k] = 1
            else:
                keyfeatures[k] += 1
keyfeatures = {k: v for k, v in keyfeatures.items() if v > 1}
sorted_x = sorted(keyfeatures.items(), key=operator.itemgetter(1),
                  reverse=True)  # dictionary with all features that are mentioned in at least two webshop products, sorted based on frequency

sth = th_keymatching
sim = {}  # dictionary with all feature types and the various representations in which they occur
sim[sorted_x[0][0]] = []
th = sth
for i in range(1, len(sorted_x)):
    j = 0
    while similar(sorted_x[i][0], sorted_x[j][0], w_seqmatch, w_jacsim) < th:
        j += 1
    if j == i:
        sim[sorted_x[i][0]] = []
    else:
        if sorted_x[j][0] in sim.keys():
            sim[sorted_x[j][0]].append(sorted_x[i][0])
            th = sth
        else:
            sth -= 0.1
            i -= 1
simcut = {k: v for k, v in sim.items() if
          len(v) >= 1}  # dictionary with all feature types that have at least two different representations in the Features Maps


def replacekeys(text):
    for key in simcut.keys():
        for value in simcut[key]:
            text = text.replace(value, key)
    return text


def cleankey(text):
    text = replacekeys(text)
    text = text.lower()
    text = text.replace(' ', '')
    return text


### Function to find potential modelID in product titles
def findmodelID(text):
    """
    Finds potential modelID in the title of the webshop products
    """
    text = text.replace('(', '')
    text = text.replace(')', '')

    textlist = ' '.join(w for w in text.split() if any(x.isdigit() for x in w)).split()
    maxlenword = ''
    for word in textlist:
        len1 = len(maxlenword)
        len2 = len(word)
        if len2 > len1:
            maxlenword = word
    if len(maxlenword) < minlength_mid:
        return 'None'
    else:
        return maxlenword


### Create dataframe for WLSH
shop1 = 'amazon.com'
shop2 = 'newegg.com'
shop3 = 'bestbuy.com'
shop4 = 'thenerds.net'
shops = [shop1, shop2, shop3, shop4]
cleanShops = [cleanshop(item) for item in shops]  #

brandnames = {}  # dictionary with all brands and how often they occur
for key in data.keys():
    for i in range(len(data[key])):
        for k in data[key][i]['featuresMap'].keys():
            if k == 'Brand' or k == 'Brand Name' or k == 'Brand Name:':
                if data[key][i]['featuresMap'][k] not in brandnames.keys():
                    brandnames[data[key][i]['featuresMap'][k]] = 1
                else:
                    brandnames[data[key][i]['featuresMap'][k]] += 1

DListOfBrands = brandnames.keys()
ListOfBrands = [cleanbrand(item) for item in DListOfBrands]  # list of all brands that occur in dataset

# Create Dataset and clean content
DataSet = pd.DataFrame(columns=['key', 'potmodelID', 'title', 'shop',
                                'kvp'])  # Dataframe with 1624 webshop products and columns for key (modelID), potential modelID from title, shop, and modelwords from title and Feature Maps
keys = []
titles = []
shop = []
kvps = []
potmodelID = []
for key in data.keys():
    for i in range(len(data[key])):
        keys.append(key)
        potmodelID.append(findmodelID(data[key][i]['title']))
        titles.append(cleantitle(data[key][i]['title']).split())
        shop.append(cleanshop(data[key][i]['shop']))
        kvpi = []
        for kvp in data[key][i]['featuresMap']:
            if kvp == 'Brand' or kvp == 'Brand Name' or kvp == 'Brand Name:':
                value = cleanbrand(data[key][i]['featuresMap'][kvp])
                kvpi.append(value)
            if replacekeys(kvp) not in simcut.keys():
                continue
            else:
                k = cleankey(kvp)
                value = cleanvalue(data[key][i]['featuresMap'][kvp])
                kvpi.append(k + ':' + value)
        kvps.append(kvpi)

DataSet['key'] = keys
DataSet['potmodelID'] = potmodelID
DataSet['title'] = titles
DataSet['shop'] = shop
DataSet['kvp'] = kvps

del keys, titles, shop, kvps

# Create dataframe for plot information
plot_info = pd.DataFrame(columns=['Frac of comp', 'PQ', 'PC', 'F1*', 'Fbeta', 'Precision', 'Recall', 'F1'])
FOCv = []
PQv = []
PCv = []
F1_starv = []
Fbetav = []
Precisionv = []
Recallv = []
F1v = []
### Start of bootstrapping
boot_it = 5
print('Now starting ' + str(boot_it) + ' bootstrap iterations')
for i in range(boot_it):
    print("|------------------------------------------ Bootstrap " + str(
        i + 1) + " ------------------------------------------|")
    indices_keep = []
    for j in range(len(DataSet)):
        rand = random.randint(1, len(DataSet))
        if rand not in indices_keep:
            indices_keep.append(rand)
    indices_delete = []
    for k in range(len(DataSet)):
        if k not in indices_keep:
            indices_delete.append(k)
    DataSetB = DataSet.drop(indices_delete)
    DataSetB = DataSetB.reset_index(drop=True)
    print("Created bootstrap DataSet with " + str(len(DataSetB)) + " randomly selected products, which is " + str(
        round((len(DataSetB) / len(DataSet)) * 100)) + '% of original DataSet')


    #%% md
    ### Define supporting functions for WLSH
    #%%
    def progressBar(name, value, endvalue, bar_length=50, width=20):
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length) - 1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        sys.stdout.write("\r{0: <{1}} : [{2}]{3}%".format(name, width, arrow + spaces, int(round(percent * 100))))


    def findrbp(n, r):
        b = 1
        while r * b <= n * share_LSH:
            b += 1
        p = r * b
        return p, r, b


    #%% md
    ### Compute dictionary with model words, remove shop names from this dictionary, and weight brand and potential model ID
    #%%
    WordCount = {}  # dictionary with all modelwords and how often they occur
    for title in DataSetB['title']:
        for word in title:
            if word not in WordCount.keys():
                WordCount[word] = 1
            else:
                WordCount[word] += 1
    del word, title

    for kvp in DataSetB['kvp']:
        for word in kvp:
            if word not in WordCount.keys():
                WordCount[word] = 1
            else:
                WordCount[word] += 1
    del word, kvp

    wordstoremove = cleanShops
    for item in wordstoremove:
        if item in WordCount.keys():
            del WordCount[item]
    del item, wordstoremove

    WordCount = {k: v for k, v in WordCount.items() if v >= 2}  # retain only modelwords that occur more than once
    inputWords = list(WordCount.keys())  # list of all modelwords
    print('Original list of inputwords contains ' + str(len(inputWords)) + ' inputwords')

    count_potmid = 0
    count_brand = 0

    for i in range(w_brand - 1):
        for brand in ListOfBrands:
            inputWords.append(brand)
    for i in range(w_potmid - 1):
        for mid in potmodelID:
            inputWords.append(mid)

    print('Final list of inputwords contains ' + str(
        len(inputWords)) + ' inputwords after weighting of Brand en Potential Model IDs')

    #%% md
    ### Create binary input matrix for LSH
    #%%
    products = DataSetB.index
    inputMatrix = np.zeros((len(inputWords), len(products)))  # binary input matrix for LSH

    for p in range(inputMatrix.shape[1]):
        progressBar("Creating input matrix    ", p, inputMatrix.shape[1] - 1)
        title = DataSetB['title'][p]
        kvp = DataSetB['kvp'][p]
        for word_i in range(inputMatrix.shape[0]):
            if inputWords[word_i] in title:
                inputMatrix[word_i][p] = 1
    del kvp, title, word_i


    #%% md
    ### Create signature matrix for WLSH using random hash-functions
    #%%
    def isPrime(x):
        for l in range(2, int(x ** 0.5) + 1):
            if x % l == 0:
                return False
        return True


    def findPrimeNum(num):
        for m in range(num, 10000, 1):
            if isPrime(m):
                return m


    r = rpbLSH
    (permutations, rows, bands) = findrbp(len(inputMatrix), r)

    th = pow((1 / bands), (1 / rows))

    print('')
    print('Number of permutations    :  ' + str(permutations))
    print('Number of rows per band   :  ' + str(rows))
    print('Number of bands           :  ' + str(bands))

    SignatureMatrix = np.ones((permutations, len(products))) * 1000000
    a = random.sample(range(inputMatrix.shape[0]), permutations)
    b = random.sample(range(inputMatrix.shape[0]), permutations)


    def hash(a, b, r):
        return (int(a * r + b)) % findPrimeNum(permutations)


    def hash_factory(n):
        return lambda x: hash(a[n], b[n], x)


    hashes = [hash_factory(_) for _ in range(permutations)]  # a list of hash functions

    for r in range(inputMatrix.shape[0]):
        progressBar("Creating signature matrix", r, inputMatrix.shape[0] - 1)
        hashvalues = []
        for permutation in range(permutations):
            hashvalues.append(hashes[permutation](r))
        for p in range(inputMatrix.shape[1]):
            if inputMatrix[r][p] == 1:
                for permutation in range(permutations):
                    SignatureMatrix[permutation][p] = min(hashvalues[permutation], SignatureMatrix[permutation][p])
    #%% md
    ### Perform Weighted Locality Sensitive Hashing (WLSH)
    #%%
    reps = repsLSH
    buckets = {}
    for r in range(reps):
        shuff = SignatureMatrix.copy()
        np.random.shuffle(shuff)
        for p in range(shuff.shape[1]):
            for b in range(bands):
                h = str(reps) + ' ' + str(b) + ' ' + (str([round(item) for item in shuff[:, p][b:b + rows]]))
                if h not in buckets.keys():
                    buckets[h] = [p]
                else:
                    if p not in buckets[h]:
                        buckets[h].append(p)

    buckets = {k: v for k, v in buckets.items() if len(v) >= 2}

    del reps, p, b, h, shuff, r

    #%% md
    ### Define candidate pairs from buckets
    #%%
    candidates = {}
    for key in buckets.keys():
        for comb in list(combinations(buckets[key], 2)):
            if comb[0] > comb[1]:
                comb = tuple([comb[1], comb[0]])
            if comb not in candidates.keys():
                candidates[comb] = 1
            else:
                candidates[comb] += 1

    del key, comb

    before1 = len(candidates.keys())
    del_keys = []
    for key in candidates.keys():
        i = key[0]
        j = key[1]
        if DataSetB['shop'][i] == DataSetB['shop'][j]:
            del_keys.append(key)

    for key in del_keys:
        del candidates[key]
    after1 = len(candidates.keys())
    print('Removed ' + str(before1 - after1) + ' out of ' + str(
        before1) + ' candidate pairs due to same shop. Currently ' + str(after1) + " candidate pairs left.")

    before2 = len(candidates.keys())
    product_brand = {}
    for i in DataSetB.index:
        product_brand[i] = 'none'
        for brand in ListOfBrands:
            if brand in DataSetB['title'][i] or brand in DataSetB['kvp'][i]:
                product_brand[i] = brand
                break

    product_withbrand = {k: v for k, v in product_brand.items() if v != 'none'}
    del_keys = []
    for key in candidates.keys():
        brand_i = product_brand[key[0]]
        brand_j = product_brand[key[1]]
        if brand_i == 'none' or brand_j == 'none':
            continue
        else:
            if brand_i != brand_j:
                del_keys.append(key)

    for key in del_keys:
        del candidates[key]
    after2 = len(candidates.keys())
    print('Removed ' + str(before2 - after2) + ' out of ' + str(
        before2) + ' candidate pairs due to brand inequality. Currently ' + str(after2) + " candidate pairs left.")
    #%% md
    ### Find true pairs
    #%%
    listoftruepairs = []
    real = 0
    for i in list(DataSetB.index):
        for j in list(DataSetB.index):
            if i >= j:
                continue
            else:
                if DataSetB['key'][i] == DataSetB['key'][j]:
                    pair = tuple([i, j])
                    listoftruepairs.append(pair)
                    real += 1

    #%% md
    ### Determine performance of WLSH
    #%%
    # real = 0
    # l = []
    # for key in data.keys():
    #     index = len(data[key])
    #     if index == 1:
    #         continue
    #     if index == 2:
    #         real += 1
    #         l.append(key)
    #     if index == 3:
    #         real += 3
    #         l.append(key)
    #     if index == 4:
    #         real += 6
    #         l.append(key)

    possible = {}

    counter = 0
    x = []
    for candidate in list(candidates.keys()):
        if DataSetB['key'][candidate[0]] == DataSetB['key'][candidate[1]]:
            counter += 1
            x.append(DataSetB['key'][candidate[0]])
        else:
            continue

    totalcombinationspossible = sum(np.arange(0, len(DataSetB) + 1, 1).tolist())
    print('|------------------Performance of WLSH------------------|')
    FOC = len(candidates) / totalcombinationspossible
    PQ = counter / len(candidates)
    PC = counter / real
    F1_star = (2 * PQ * PC) / (PQ + PC)
    Fbeta = (1 + beta * beta) * (PQ * PC) / (beta * beta * PQ + PC)
    print('Total number of pairs present in DataSetB    =  ' + str(real))
    print('Original pairs found                         =  ' + str(counter))
    print('Original pairs lost                          =  ' + str(real - counter))
    print('Candidates to be compaired                   =  ' + str(len(candidates)))
    print('')
    print("LSH threshold is " + str(th))
    print(str(counter) + ' out of ' + str(real) + " duplicates found")
    print(str(counter) + ' out of ' + str(len(candidates)) + " candidates are duplicates")
    print('')
    print('PQ        = ' + str(PQ))
    print('PC        = ' + str(PC))
    print('Fbeta     = ' + str(Fbeta))
    print('F1       = ' + str(F1_star))


    #%% md
    ### Perform final clustering on remaining candidate pairs
    #%%
    def jacsim(a, b):
        v1 = SignatureMatrix[:, a]
        v2 = SignatureMatrix[:, b]
        v3 = v1 - v2
        v3[v3 != 0] = 1
        jdis = sum(v3) / len(v3)
        return 1 - jdis


    DistanceMatrix = np.ones((len(products), len(products))) * 10000000
    for i in range(DistanceMatrix.shape[0]):
        for j in range(DistanceMatrix.shape[1]):
            if tuple([i, j]) in candidates.keys():
                DistanceMatrix[i][j] = 1 - jacsim(i, j)

    clustering = AgglomerativeClustering(affinity='precomputed', linkage='single', distance_threshold=th_clust,
                                         n_clusters=None).fit_predict(DistanceMatrix)

    buckets_cl = {}
    for i in range(len(clustering)):
        if clustering[i] not in buckets_cl.keys():
            buckets_cl[clustering[i]] = [i]
        else:
            buckets_cl[clustering[i]].append(i)
    buckets_cl = {k: v for k, v in buckets_cl.items() if len(v) >= 2}
    result = []
    for key in buckets_cl.keys():
        for comb in list(combinations(buckets_cl[key], 2)):
            if comb[0] > comb[1]:
                comb = tuple([comb[1], comb[0]])
            result.append(comb)
    #%% md
    ### Determine performance of clustering
    #%%
    print('|---------------Performance of clustering---------------|')
    TP = set(result).intersection(set(listoftruepairs))
    FP = set(result) - set(listoftruepairs)
    FN = set(listoftruepairs) - set(result)

    Precision = len(TP) / (len(TP) + len(FP))  # PQ
    Recall = len(TP) / (len(TP) + len(FN))  # PC

    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    print('Precision = ' + str(Precision))
    print('recall    = ' + str(Recall))
    print('F1        = ' + str(F1))

    FOCv.append(FOC)
    PQv.append(PQ)
    PCv.append(PC)
    F1_starv.append(F1_star)
    Fbetav.append(Fbeta)
    Precisionv.append(Precision)
    Recallv.append(Recall)
    F1v.append(F1)

plot_info['Frac of comp'] = FOCv
plot_info['PQ'] = PQv
plot_info['PC'] = PCv
plot_info['F1*'] = F1_starv
plot_info['Fbeta'] = Fbetav
plot_info['Precision'] = Precisionv
plot_info['Recall'] = Recallv
plot_info['F1'] = F1v
print("Fraction of comparison is " + str(FOC))
print(plot_info)

end_time = time.time()
print("Total runtime for program is " + str(round((end_time - start_time), 1)) + " seconds")