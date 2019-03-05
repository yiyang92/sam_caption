from collections import Counter


class Dictionary():
    def __init__(self, vocab_min=1):
        """
        Args:
            vocab_min: keep occured minimum times
        """
        # sentences - array of sentences
        self._vocab_min = vocab_min
        self._word2idx = {}
        self._idx2word = {}
        self._words = []

    def fit(self, text_data):
        # Tokenize
        for sentence in text_data:
            sentence = self.__tokenize(sentence)
            self._words.extend(sentence)
        self.__build_vocabulary()
    
    def transform(self, data, add_start_end=False, max_len=None):
        data_ = []
        if isinstance(data, str):
            if len(data.split(" ")) == 1:
                return self._word2idx.get(data.lower(), self._unk_toked_idx)
            else:
                data = self.__tokenize(data)
                return [
                    self._word2idx.get(wd, 
                    self._unk_toked_idx) for wd in data]
        else:
            for sentence in data:
                sentence = self.__tokenize(sentence)
                if max_len is not None:
                    if len(sentence) > max_len - 2:
                        sentence = sentence[:max_len-2]
                if add_start_end:
                    sentence = ["<BOS>"] + sentence + ["<EOS>"]
                sentence = [
                    self._word2idx.get(wd, 
                    self._unk_toked_idx) for wd in sentence]
                data_.append(sentence)
        return data_

    def __tokenize(self, sentence):
        return [word.lower().strip() for word in sentence.split(" ")]

    def __build_vocabulary(self):
        counter = Counter(self._words)
        print(counter.most_common(5))
        # words, that occur less than 5 times dont include
        sorted_dict = sorted(counter.items(), key= lambda x: (-x[1], x[0]))
        # keep n words to be included in vocabulary
        self._words = [wd for wd, count in sorted_dict
                    if count >= self._vocab_min]
        self.unk_token = "<UNK>"
        special_words = ["<PAD>", self.unk_token, "<BOS>", "<EOS>"]
        self._idx2word = {idx: wd for idx, wd in enumerate(
            special_words + self._words)}
        self._word2idx = {wd: idx for idx,wd in self._idx2word.items()}
        self._unk_toked_idx = self._word2idx[self.unk_token]
        print("Vocabulary size: ", len(self._word2idx))

    def __len__(self):
        return len(self._idx2word)
    
    @property
    def word2idx(self):
        return self._word2idx
    
    @property
    def idx2word(self):
        return self._idx2word
    
    @property
    def vocab_size(self):
        return len(self._word2idx)