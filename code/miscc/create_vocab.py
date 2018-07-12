import pickle
import os
import nltk
from collections import Counter

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not (word in self.word2idx):
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return self.idx


def create_CUB_vocab(data_dir, caption_filenames, vocab_path, threshold=1):
    """Creates a vocabulary for the CUB dataset.

    """

    # def load_captions(caption_name):  # self,
    #     cap_path = caption_name
    #     with open(cap_path, "r") as f:
    #         captions = f.read().decode('utf8').split('\n')
    #     captions = [cap.replace("\ufffd\ufffd", " ")
    #                 for cap in captions if len(cap) > 0]
    #     return captions

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adding words from each caption
    counter = Counter()
    lens = []

    for i, key in enumerate(caption_filenames):
        caption_name = '%s/text/%s.txt' % (data_dir, key)

        with open(caption_name, "r") as f:
            captions = f.read().decode('utf8').split('\n')

        for caption in captions:
            if len(caption) > 0:
                caption = caption.replace("\ufffd\ufffd", " ")
                tokens = nltk.tokenize.word_tokenize(caption.lower())
                lens.append(len(tokens))
                counter.update(tokens)

        print('[%d/%d] Creating vocabulary.' %(i, len(caption_filenames)))

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    for word in words:
        vocab.add_word(word)

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    import numpy
    print('Max Len: ', max(lens))
    print('Average Len: ', numpy.mean(lens))
    print('Created vocabulary. Total items are: %d' %(len(vocab)))
    return vocab


def create_FLOWER_vocab(data_dir, caption_filenames, vocab_path, class_id, threshold=1):
    """Creates a vocabulary for the CUB dataset.

    """

    # def load_captions(caption_name):  # self,
    #     cap_path = caption_name
    #     with open(cap_path, "r") as f:
    #         captions = f.read().decode('utf8').split('\n')
    #     captions = [cap.replace("\ufffd\ufffd", " ")
    #                 for cap in captions if len(cap) > 0]
    #     return captions

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adding words from each caption
    counter = Counter()
    lens = []

    for i, key in enumerate(caption_filenames):
        caption_name = '%s/text/class_%05d/%s.txt' % (data_dir, class_id[i], key.split('/')[1])

        with open(caption_name, "r") as f:
            captions = f.read().decode('utf8').split('\n')

        for caption in captions:
            if len(caption) > 0:
                caption = caption.replace("\ufffd\ufffd", " ")
                tokens = nltk.tokenize.word_tokenize(caption.lower())
                lens.append(len(tokens))
                counter.update(tokens)

        print('[%d/%d] Creating vocabulary.' %(i, len(caption_filenames)))

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    for word in words:
        vocab.add_word(word)

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    import numpy
    print('Max Len: ', max(lens))
    print('Average Len: ', numpy.mean(lens))
    print('Created vocabulary. Total items are: %d' %(len(vocab)))
    return vocab

def load_filenames(data_dir):
    filepath = os.path.join(data_dir, 'train', 'filenames.pickle')
    with open(filepath, 'rb') as f:
        filenames = pickle.load(f)
    print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
    return filenames


if __name__ == '__main__':
    data_dir = '../data/birds'
    filenames = load_filenames(data_dir)
    vocab_path = os.path.join(data_dir, 'cub_vocab.pkl')
    create_CUB_vocab(data_dir, filenames, vocab_path)