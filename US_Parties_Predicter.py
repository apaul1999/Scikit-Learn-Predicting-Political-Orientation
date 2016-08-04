# Author: Anamitra Paul <p_anamitra@yahoo.com>
# Disclaimer: Some minor portions of code that are not my creation have been borrowed from the scikit-learn.org website

from __future__ import print_function

import pandas
from sklearn import cross_validation
from sklearn import metrics
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

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.02, random_state=0)

    # Tokenization and Extraction

    transformer = Pipeline([
        ('vectorizer', HashingVectorizer(stop_words='english', non_negative=True, ngram_range=(1, 3),
                                         analyzer='word', norm='l2')),
        ('tfidf', TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)),
    ])

    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)
    docs = ['Crooked Hillary', 'Bernie Sanders for president', 'Climate change is real', 'Reduce immigration',
            'Abortion should be illegal']

    X_new = transformer.transform(docs)


    # Prediction

    def predict(clf):
        print()
        print('#' * 100)
        print(clf, '\n')
        clf.fit(X_train, y_train)
        y_pred_classification = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred_classification)
        print('Accuracy: %0.3f \n' % accuracy)
        y_pred = clf.predict(X_new)
        for i in range(0, len(docs)):
            print('\"%s\" is a comment likely made by a %s supporter.' % (docs[i], parties[y_pred[i]]))


    results = []

    # Predicting with a Linear Support Vector Machine
    results.append(predict(LinearSVC(loss='squared_hinge', penalty='l2', dual=True, tol=1e-4)))

    # Predicting with a Multinomial Naive Bayes classifier
    results.append(predict(MultinomialNB(alpha=0.01, fit_prior=True)))
