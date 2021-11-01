from sklearn.feature_extraction.text import CountVectorizer
corpus = [   
   'apple ball cat',
   'ball cat dog',
   ]
# corpus = [   
#    'apple ball cat cat',
#    'ball cat dog ',
#    ]
# corpus = [   
#    'zebra apple ball cat cat',
#    'ball cat dog elephant',
#    'very very unique'
#    ]
# corpus = [
#     'This is the first document.',
#     'This document is the second document.',
#     'And this is the third one.',
#     'Is this the first document?',
# ]
vectorizer = CountVectorizer()
X1= vectorizer.fit_transform(corpus)
vectorizer.get_feature_names_out()
# print(vectorizer.get_feature_names_out())
# print(X1.toarray())
# print(X1)
max_features= 100
ngrams = 3
vectorizer2= CountVectorizer(max_features=max_features, ngram_range=(1,ngrams))
X2=vectorizer2.fit_transform(corpus)
print(vectorizer2.get_feature_names_out())
print(X2.toarray())

# vectorizer3= CountVectorizer(stop_words="engilsh", max_features=max_features, ngram_range=(1,ngrams))
# X3=vectorizer3.fit_transform(corpus)
# print(vectorizer3.get_feature_names_out())
# print(X3.toarray())