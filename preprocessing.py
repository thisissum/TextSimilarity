import gensim
import numpy as np 
import os
import json


def load_jsonl(path, encoding='utf-8'):
    with open(path,'r',encoding=encoding) as f:
        data = list(map(lambda x:json.loads(x),f.readlines()))
    return data


def preprocessing(data, data_path, emb_path=None):
    sentences1 = list(map(lambda x:x['sentence1'].split(), data))
    sentences2 = list(map(lambda x:x['sentence2'].split(), data))
    targets = list(map(lambda x:x['gold_label'], data))

    if emb_path is not None:
        #simply use word2vec embeddings
        w2v_model = gensim.models.Word2Vec(
            sentences1 + sentences2, size=200, iter=10, min_count=1)
        #build word2id and emb_matrix
        word2id = {'<pad>':0, '<unk>':1}
        emb_matrix = np.zeros((len(w2v_model.wv.vocab)+2, w2v_model.vector_size))
        for word in w2v_model.wv.vocab.keys():
            ind = len(word2id)
            word2id[word] = ind
            emb_matrix[ind,:] = w2v_model.wv[word]
        
        #save all
        with open(os.path.join(emb_path, 'word2id.json'), 'w', encoding='utf-8') as f1:
            json.dump(word2id,f1)

        np.save(os.path.join(emb_path, 'w2v_embedding.npy'), emb_matrix)

    with open(os.path.join(data_path, 'sentences1.json'), 'w', encoding='utf-8') as f2:
        json.dump(sentences1, f2)
    with open(os.path.join(data_path, 'sentences2.json'), 'w', encoding='utf-8') as f3:
        json.dump(sentences2, f3)
    with open(os.path.join(data_path, 'targets.json'), 'w', encoding='utf-8') as f4:
        json.dump(targets, f4)
    print('================================== DONE ==================================')


if __name__ == '__main__':
    data = load_jsonl('./data/snli_1.0_train.jsonl')
    preprocessing(data, './data/train', './embedding')
    data = load_jsonl('./data/snli_1.0_dev.jsonl')
    preprocessing(data, './data/dev')
    data = load_jsonl('./data/snli_1.0_test.jsonl')
    preprocessing(data, './data/test')