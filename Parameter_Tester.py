# Author: Anamitra Paul <p_anamitra@yahoo.com>
# Disclaimer: Some minor portions of code that are not my creation have been borrowed from the scikit-learn.org website

from __future__ import print_function

import pandas
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

if __name__ == '__main__':

    path = 'C:\\Users\Anamitra Paul\Downloads\Political Data and Classifiers\\US_Political_Parties_Dataset.csv'
    usp = pandas.read_csv(path, header=0, names=['Content', 'Party_Num'])

    X = usp.Content
    y = usp.Party_Num

    pipeline = Pipeline([
        ('vect', HashingVectorizer(stop_words='english', non_negative=True, ngram_range=(1, 3), analyzer='word',
                                   norm='l2')),
        ('tfidf', TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)),
        ('clf', LinearSVC(dual=True)),
    ])

    print(pipeline.get_params().keys())

    parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),
                  'clf__loss': ('squared_hinge', 'hinge'), 'clf__tol': (1e-4, 1e-3)}

    gs = GridSearchCV(pipeline, parameters, n_jobs=-1)
    gs = gs.fit(X, y)
    best_parameters, accuracy, _ = max(gs.grid_scores_, key=lambda x: x[1])

    for parameter in sorted(parameters.keys()):
        print('%s: %r' % (parameter, best_parameters[parameter]))

    print(accuracy)
