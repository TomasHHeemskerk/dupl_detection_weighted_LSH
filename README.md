# Duplicate Detection using Weighted Locality Sensitive Hashing (WLSH)
This repository describes a duplicate detection algorithm that makes use of pre-processing steps (cleaning & reading of data) and a Weighted Locality Sensitive Hashing (WLSH) procedure.

## Data
The data used comes from 1624 TV products from 4 different webshops. Relevant data points include modelID, shop, product title and a features map with product descriptions

## Hyperparameters
Contains several hyperparameters for data reading, weighting of LSH, and agglomerative clustering algorithm.

## Data cleaning functions
Contains several functions that clean and standardize representations of product title and key-value pairs in product description.

## Data reading functions
Contains several functions that read & match keys from product description and concatenate with adjacent values. Also contains function to find potential modelIDs in the product titles.

## DataSet generation
Creates DataSet containing for each product: modelID, potential modelID, title, shop, and key-value pairs (extracted from product description). Also creates Plot Information Dataframe to gather datapoints needed for plotting algorithm performance.

## Bootstrapping
Draws N products with replacement to fill test set and includes a draw only if the product is not yet in the test set. In total, 5 bootstraps are performed.

## Supporting functions for WLSH
Contains a progress bar function and a function that finds the correct number of permutations and bands, given the number of rows per band and the size of the input matrix.

## Dictionary generation
Creates dictionary with model words, removes shop names and weights model words of type Brand and Potential ModelID

## Binary Input Matrix generation
Creates binary input matrix by 'gluing' the binary vector representations of all products.

## Signature Matrix generation
Creates signature matrux for WLSH using minhashing with random hash-functions.

## WLSH execution
Buckets all products using the signature matrix.

## Candidate duplicates detection & cleaning
Finds potential duplicate pairs by linking all products for each seperate bucket, and removes pairs with same shop or different brand.

## Performance calculation WLSH
Calculates performance measures of WLSH: PC, PQ, F1_star and F_beta.

## Final clustering
Deploys final agglomerative clustering method on all potential duplicate pairs.

## Performance calculation clustering
Calculcates performance measures of clustering: Precision, Recall and F1.

