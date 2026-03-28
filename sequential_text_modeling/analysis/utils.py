import math
import numpy as np
from collections import Counter
from transformers import BertTokenizer
from IPython.display import display, HTML

data_path = 'path_to_ap_news_data'
mpm_logs = 'path_to_mpm_logs/'

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-cased', clean_up_tokenization_spaces=False)

def to_word(token):
    return tokenizer.convert_ids_to_tokens(token)

def to_sentence(tokens, join_words=False):
    tokens = to_word(tokens)
    if not join_words:
        return ' '.join(tokens)
    words = []
    for t in tokens:
        if t.startswith('##'):
            words[-1] += t[2:]
        else:
            words.append(t)
    return ' '.join(words)

def topT_words(words, T=10):
    counter = Counter(words)
    topT = counter.most_common(T)
    return [tup[0] for tup in topT]

def compute_tfidf(corpus):
    N = len(corpus)

    # Calculate term frequencies (TF)
    tf_list = []
    for doc in corpus:
        tf = {}
        total_terms = len(doc)
        for word in doc:
            tf[word] = tf.get(word, 0) + 1
        for word in tf:
            tf[word] /= total_terms
        tf_list.append(tf)

    # Calculate document frequencies (DF)
    df = {}
    for doc in corpus:
        unique_terms = set(doc)
        for word in unique_terms:
            df[word] = df.get(word, 0) + 1

    # Step 3: Calculate IDF
    idf = {}
    for word, doc_freq in df.items():
        idf[word] = math.log(N / (1 + doc_freq))

    # Calculate TF-IDF (returning (tfidf, word) tuples)
    tfidf_corpus = []
    for tf in tf_list:
        doc_tfidf = []
        for word in tf:
            tfidf = tf[word] * idf[word]
            doc_tfidf.append((tfidf, word))
        tfidf_corpus.append(doc_tfidf)

    return tfidf_corpus

def get_topics(seed):
    root_dir = f'{mpm_logs}/100topics/beta=0.1-seed={seed}'
    results = np.load(f'{root_dir}/predictions/results.npz')
    data = np.load(data_path)
    tokens = data['tokens']
    masks = data['masks']
    word_starts = data['word_starts']
    ids = results['doc_ids']
    attns = results['attns']
    slots = results['slots']
    
    K = 5
    slot_words = []
    M = attns.shape[1]
    for i in range(len(ids)):
        slot_words.append([])
        for k in range(K):
            slot_words[-1].append([])
        for j in range(M):
            if word_starts[ids[i]][j]:
                word = to_word(int(tokens[ids[i]][j]))
            else:
                word += to_word(int(tokens[ids[i]][j]))[2:]
            if j == M-1 or word_starts[ids[i]][j+1]:
                qz = attns[i][j]
                if masks[ids[i]][j] == 0:
                    assert sum(qz) == 0
                    break
                k = np.argmax(qz)
                assert not word.startswith('##')
                slot_words[-1][k].append(word)

    mixing = np.load(f'{root_dir}/predictions/mixing.npy')

    N, K, L = mixing.shape
    global_cluster_words = []
    for _ in range(L):
        global_cluster_words.append([])

    gc_idxs = np.argmax(mixing, axis=2)
    for i in range(N):
        for k in range(K):
            global_cluster_words[gc_idxs[i][k]] += slot_words[i][k]
    global_tfidf_words = compute_tfidf(global_cluster_words)

    T = 10
    for ell in range(L):
        sorted_words = global_tfidf_words[ell]
        sorted_words.sort()
        sorted_words = [tup[1] for tup in sorted_words[::-1]]

    K = 10
    topics = []
    for ell in range(L):
        sorted_words = global_tfidf_words[ell]
        sorted_words.sort()
        sorted_words = [tup[1] for tup in sorted_words[::-1]]
        topics.append(sorted_words[:K])
    return topics