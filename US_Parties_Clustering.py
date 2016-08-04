# Author: Anamitra Paul <p_anamitra@yahoo.com>
# Disclaimer: Some minor portions of code that are not my creation have been borrowed from the scikit-learn.org website

from __future__ import print_function

import warnings

import numpy
import pandas
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

if __name__ == '__main__':

    # Importing Data

    path = 'C:\\Users\Anamitra Paul\Downloads\Political Data and Classifiers\\US_Political_Parties_Dataset.csv'
    usp = pandas.read_csv(path, header=0, names=['Content', 'Party_Num'])

    # Party_Num(0: Democratic Party, 1: Republican Party)

    parties = [
        'Democrat',
        'Republican',
    ]

    print('US Political Parties: ')
    print(parties)

    X = usp.Content
    y = usp.Party_Num

    # Pipelining and Vectorization

    num_samples = len(X)
    num_clusters = numpy.unique(y).size

    # Use the commented pipeline if you require low memory usage

    '''pipeline = Pipeline([
        ('vect', HashingVectorizer(stop_words='english', non_negative=True, ngram_range=(1, 3), analyzer='word',
                                   norm='l2')),
        ('tfidf', TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)),
    ])'''

    pipeline = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), analyzer='word', norm='l2')

    print(pipeline.get_params())

    X = pipeline.fit_transform(X)

    print('%d samples' % num_samples)
    print('%d categories' % num_clusters)
    print('%d features' % X.shape[1])

    # Clustering

    warnings.filterwarnings("ignore", category=DeprecationWarning)


    def evaluate(clust):
        print()
        print(clust, '\n')
        clust.fit(X)

        print('Clustering Inertia: %0.3f' % clust.inertia_)
        print('V-Measure: %0.3f' % metrics.v_measure_score(y, clust.labels_))
        print('Adjusted Rand-Index: %.3f' % metrics.adjusted_rand_score(y, clust.labels_))
        print('Cluster Labels: \n', clust.labels_)
        print()
        try:
            sorted_centroids = clust.cluster_centers_.argsort()[:, ::-1]
            words = pipeline.get_feature_names()
            for i in range(num_clusters):
                print('Cluster %d:' % i, end='')
                for j in sorted_centroids[i, :10]:
                    print(' %s' % words[j], end='')
                print()
        except:
            print()
            print('Cannot retrieve the top feature names. You are probably using a HashingVectorizer.')


    results = []

    # Clustering with a K-Means Algorithm
    results.append(evaluate(KMeans(n_clusters=num_clusters, max_iter=1000, precompute_distances=True, n_jobs=-1,
                                   n_init=100)))

    # Clustering with a Mini-Batch K-Means Algorithm
    results.append(evaluate(MiniBatchKMeans(n_clusters=num_clusters, max_iter=1000, batch_size=100, compute_labels=True,
                                            n_init=100, reassignment_ratio=1.0)))
