from nltk import word_tokenize
from nltk.corpus import reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")


def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text));
    words = [word for word in words if word not in cachedStopWords]
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)));
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens));
    return filtered_tokens


# Return the representer, without transforming
def tf_idf(docs):
    tfidf = TfidfVectorizer(tokenizer=tokenize, max_df=0.90, max_features=3000, use_idf=True,
                            sublinear_tf=True);
    tfidf.fit(docs);
    return tfidf;


def feature_values(doc, representer):
    doc_representation = representer.transform([doc])
    features = representer.get_feature_names()
    return [(features[index], doc_representation[0, index]) for index in doc_representation.nonzero()[1]]

def get_tf(docs,vocab):
    tf=CountVectorizer(tokenizer=tokenize,vocabulary= vocab);
    temp=tf.fit_transform(docs)
    return temp


def collection_stats():
    # List of documents
    documents = reuters.fileids()
    print(str(len(documents)) + " documents");

    train_docs = list(filter(lambda doc: doc.startswith("train"), documents));
    print(str(len(train_docs)) + " total train documents");

    test_docs = list(filter(lambda doc: doc.startswith("test"), documents));
    print(str(len(test_docs)) + " total test documents");

    # List of categories
    categories = reuters.categories();
    print(str(len(categories)) + " categories");

    # Documents in a category
    category_docs1 = reuters.fileids("wheat");
    category_docs2 = reuters.fileids("grain");
    # Words for a document
    document_id = category_docs1[0]
    document_words = reuters.words(category_docs1[0]);

    print(document_words);

    # Raw document
    print(reuters.raw(document_id));


def main():
    #train_docs = []
    test_docs = []
    train_docs=['He watches basketball and baseball', 'Julie likes to play basketball', 'Jane loves to play baseball']
    collection_stats()
    #for doc_id in reuters.fileids():
        #if doc_id.startswith("train"):
            #train_docs.append(reuters.raw(doc_id))
        #else:
            #test_docs.append(reuters.raw(doc_id))

    representer = tf_idf(train_docs);
    vocab=representer.vocabulary_
    print ((vocab))
    train_tf=get_tf(train_docs,vocab);
    arr=train_tf.toarray();
    print(arr)
    print("hello")


    #for doc in test_docs:
        #print(feature_values(doc, representer))


if __name__ == "__main__":
    main()
