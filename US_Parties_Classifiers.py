# Author: Anamitra Paul <p_anamitra@yahoo.com>
# Disclaimer: Some minor portions of code that are not my creation have been borrowed from the scikit-learn.org website

from __future__ import print_function

import pandas
from sklearn import cross_validation
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

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

    # Initializing Transformer

    vect = HashingVectorizer(stop_words='english', non_negative=True, ngram_range=(1, 3),
                             analyzer='word', norm='l2')
    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)

    # Scoring Input

    scorers = ['accuracy', 'average_precision', 'f1', 'precision', 'recall', 'roc_auc']
    print(scorers)
    scoring_index = input('Please input the index of the scorer you want to use: ')
    scoring = scorers[int(scoring_index)]
    print('%s selected' % scoring)


    # Classifier Benchmarks

    def evaluate(clf):
        print()
        print('#' * 100)
        print(clf, '\n')
        pipeline = Pipeline([
            ('vectorizer', vect),
            ('tfidf', tfidf),
            ('classifier', clf),
        ])
        skf = cross_validation.StratifiedKFold(y, 5)
        scores = cross_validation.cross_val_score(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        score = scores.mean()
        print('%s score: %0.3f (+/- %0.3f)' % (scoring, score, scores.std() * 2))
        clf_tostring = str(clf).split('(')[0]
        return clf_tostring, score


    results = []

    # Training a Linear Support Vector Machine
    results.append(evaluate(LinearSVC(loss='squared_hinge', penalty='l2', dual=True, tol=1e-4)))

    # Training a Multinomial Naive Bayes classifier
    results.append(evaluate(MultinomialNB(alpha=0.01, fit_prior=True)))
