import gensim, logging, os

model = gensim.models.Word2Vec.load_word2vec_format('vectors', binary=False)
print model.similar_by_word("FuncDef", topn=20)
# print model.similar_by_word("ExprList")
# print model.similar_by_word("ID")
# print model.similar_by_word("TernaryOp")
