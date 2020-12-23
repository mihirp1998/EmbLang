import pickle
import evaluate
import numpy as np
import ipdb
st =ipdb.set_trace
# val0 =  pickle.load(open("latent_mean_0.p","rb"))
# val1= pickle.load(open("latent_mean_1.p","rb"))
# sval0= pickle.load(open("sentence_0.p","rb"))

# sval1= pickle.load(open("sentence_1.p","rb"))
# print(sval0,sval1)
# print(np.argmin(evaluate.findLossMatrix(val0,val1),1))
# 7
seven =pickle.load(open("latent_mean_7.p","rb"))
sevens =pickle.load(open("sentence_7.p","rb"))

one =pickle.load(open("latent_mean_1.p","rb"))
ones = pickle.load(open("sentence_1.p","rb"))

five =pickle.load(open("latent_mean_5.p","rb"))
fives  =pickle.load(open("sentence_5.p","rb"))

print(np.argmin(evaluate.findLossMatrix(seven,one),1),sevens,ones)
# import glob
# val = glob.glob("sentence*.p")
# print(val)
# sent = [pickle.load(open(i,"rb"))for i in val]
# new = [i for i in sent if type(i)==type([])]
# # print([i for i in sent if "cube" in i])
# # st()
# for k,i in enumerate(new):
# 	for j in i:
# 		if "cube" in j:
# 			print(j,k)
