import gensim, logging, os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import random
 
# sentences = [['first', 'sentence'], ['second', 'sentence']]
# # train word2vec on the two sentences
# model = gensim.models.Word2Vec(sentences, min_count=1)

def rearrange(line):
	return line[1:len(line)/2+1] + [line[0]] + line[len(line)/2+1:]

class AbstractSyntaxTokens(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
            	if line[0].isdigit():
            		continue
            	line = line.split()
            	if len(line) > 2:
            		line = rearrange(line)
                yield line

SIZE = 100
WINDOW = 10
tokens = AbstractSyntaxTokens('ast_files') # a memory-friendly iterator
model = gensim.models.Word2Vec(tokens, size=SIZE, window=WINDOW)
try:
	model.save_word2vec_format("vectors_" + str(WINDOW) + "_" + str(SIZE) + "_inorder", fvocab="tokens", binary=False)
except:
	model.save_word2vec_format("vectors_noname", fvocab="tokens", binary=False)
# print model.similar_by_word('ExprList')

