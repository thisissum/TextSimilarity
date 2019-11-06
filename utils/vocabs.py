import json


class Vocabulary(object):
    """
    """

    def __init__(self, word2id=None, pad=0, unk=1):
        if word2id is None:
            self.word2id = {'<pad>': pad, '<unk>': unk}
        else:
            self.word2id = word2id

    def __getitem__(self, key):
        return self.word2id[key]

    def add_word(self, char):
        if char not in self.word2id.keys():
            self.word2id[char] = len(self.word2id)

    def add_sentence(self, sentence):
        """
        args:
            sentence: List[String], where len(String) == 1
        """
        for char in sentence:
            self.add_word(char)

    def get_pad(self):
        return self.word2id['<pad>']

    def get_word2id(self):
        return self.word2id

    def transform(self, word):
        if word in self.word2id.keys():
            return self.word2id[word]
        else:
            return self.word2id['<unk>']

    def transform_sentence(self, sentence):
        return list(map(lambda x: self.transform(x), sentence))

    def transfrom_sentences(self, sentences):
        return list(map(lambda sentence: self.transform_sentence(sentence), sentences))

    def save(self, path, encoding='utf-8'):
        with open(path, 'w', encoding=encoding) as f:
            json.dump(self.word2id, f)

    @classmethod
    def load(cls, path, encoding='utf-8'):
        obj = cls.__new__(cls)
        with open(path, 'r', encoding=encoding) as f:
            word2id = json.load(f)
        obj.__init__(word2id)
        return obj
