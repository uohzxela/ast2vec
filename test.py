import gensim, logging, os

VECTORS_FILE = "vectors_10_100_inorder"
TSNE_FILE = "tsne_" + "_".join(VECTORS_FILE.split("_")[1:]) + ".png"

model = gensim.models.Word2Vec.load_word2vec_format(VECTORS_FILE, binary=False)
print model.similar_by_word("While", topn=20)
print model.doesnt_match("UnaryOp BinaryOp While TernaryOp".split())
print model.doesnt_match("If For TernaryOp While Continue DoWhile".split())
print model.doesnt_match("ID Constant IdentifierType FuncDef".split())

# print model.similar_by_word("ExprList")
# print model.similar_by_word("ID")
# print model.similar_by_word("TernaryOp")


labels = []
X = []
tokens = open('tokens')
for token in tokens:
	token = token.split(" ")[0]
	labels.append(token)
	X.append(model[token])

def plot_with_labels(low_dim_embs, labels, filename=TSNE_FILE):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=5, n_components=2, init='pca', n_iter=5000)


  low_dim_embs = tsne.fit_transform(X)
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")