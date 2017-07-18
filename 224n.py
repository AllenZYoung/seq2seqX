from nltk.corpus import wordnet as wn
panda = wn.sy
hyper = lambda s:s.hypernyms()
l = list(panda.closure(hyper))
print(l)