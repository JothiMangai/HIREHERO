from sklearn.feature_extraction.text import TfidfVectorizer

def generate_top_words(token):
    tfidf = TfidfVectorizer(max_df=0.05, min_df=0.002)
    words = tfidf.fit_transform(token)
    
    # Get the indices of the words with the highest tf-idf score
    top_word_indices = words.max(axis=0).toarray()[0].argsort()[::-1][:10]
    
    # Get the vocabulary of the TfidfVectorizer
    vocab = tfidf.vocabulary_
    
    # Create a list of tuples containing the word and its tf-idf score
    top_words = [(word, words.max(axis=0).toarray()[0][vocab[word]]) for word in vocab if vocab[word] in top_word_indices]
    
    return top_words
