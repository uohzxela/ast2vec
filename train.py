import gensim, logging, os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
# sentences = [['first', 'sentence'], ['second', 'sentence']]
# # train word2vec on the two sentences
# model = gensim.models.Word2Vec(sentences, min_count=1)

class MyASTokens(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
            	if line[0].isdigit():
            		continue
                yield line.split()
 
tokens = MyASTokens('ast_files') # a memory-friendly iterator
model = gensim.models.Word2Vec(tokens)
model.save_word2vec_format("vectors", fvocab="tokens", binary=False)
# print model.similar_by_word('ExprList')