from nltk import word_tokenize
from nltk.corpus import reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from scipy.io import savemat

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
    train_docs = []
    test_docs = []
    cat=[]
    #train_docs=['He watches basketball and baseball', 'Julie likes to play basketball', 'Jane loves to play baseball']
    #collection_stats()
    word_id=dict();
    class_id=[];
    id=0;
    classes=reuters.categories();
    for catogeries in classes :
        word_id[catogeries]=id;
        class_id.append(id);
        id=id+1;
    print word_id;
    savemat('/home/shashank/class.mat',mdict={'class':classes});
    savemat('/home/shashank/class_id.mat',mdict={'class_id':class_id});
    print class_id


    train_cat = dict();
    test_cat=dict();
    train_doc_label=[];
    train_doc_id=[];
    test_doc_id=[];
    test_doc_labels=[];
    tmp_id = [];
    for doc_id in reuters.fileids():
        if doc_id.startswith("train"):
            train_doc_id.append(doc_id)
            train_docs.append(reuters.raw(doc_id))
            label = reuters.categories(doc_id)
            #print label
            for iter in label:
                id = classes.index(iter)
                tmp_id.append(id)
            #print(tmp_id)
            train_cat[doc_id]=tmp_id;
            train_doc_label.append(tmp_id)
            tmp_id = []
        else:
            test_docs.append(reuters.raw(doc_id))
            test_doc_id.append(doc_id)
            label = reuters.categories(doc_id)
            for iter in label:
                id = classes.index(iter)
                tmp_id.append(id)
            test_cat[doc_id] = tmp_id;
            test_doc_labels.append(tmp_id)
            tmp_id = []


    print(len(train_doc_label))
    savemat('/home/shashank/train_doc_id.mat', mdict={'train_doc_id': train_doc_id});
    savemat('/home/shashank/train_doc_labels.mat', mdict={'train_doc_labels': train_doc_label});
    savemat('/home/shashank/test_doc_id.mat', mdict={'test_doc_id': test_doc_id});
    savemat('/home/shashank/test_doc_labels.mat', mdict={'test_doc_labels': test_doc_labels});


    print train_cat
    print test_cat









        #for doc in test_docs:
        #print(feature_values(doc, representer))


if __name__ == "__main__":
    main()