file = "glove.6B.300d.txt"
import numpy as np
#from scipy import spatial
from numpy import dot
from numpy.linalg import norm
#from sklearn.metrics.pairwise import cosine_similarity



sim_dict_dress ={}  # dictionaries to where key is the window and values are the similar windows
sim_dict_jean = {}


def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    
     
    with open(gloveFile, encoding="utf8" ) as f:
       content = f.readlines()
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model
     
model= loadGloveModel(file) 
#print (model['hello'])


model_dress = {}  #dictionaries where key is word window and value is concatenated vector.
model_jean = {}




unk = np.zeros(300, dtype=np.float32)
i = 0
vec_dress = np.empty(shape=(21978,900))
#ls = ['hello','hi','hjjhfd']
print ("Starting with dress")
f = open("sentences_dress.txt",'r')
for lines in f.readlines():
    ls =lines.strip().split("\t")
    if (ls[1] == 'O'):
        continue


    ms = ls[0].split(" ")
    rs = []
    for items in ms:
        if items not in model:
            rs.append(unk)
        else:
            rs.append(model['items'])
    vec_dress[i] = np.concatenate((rs[0],rs[1],rs[2]), axis = None)
    model_dress[ls[0]] = vec_dress[i]
    i+=1
    #print(i)

print("done with dress")
#print (vec_dress[0])
f.close()






vec_jean = np.empty(shape=(16654,900))
i=0
print ("Starting with jean")
g = open("sentences_jean.txt",'r')
for lines in g.readlines():
    ls =lines.strip().split("\t")
    if (ls[1] == 'O'):
        continue
    ms = ls[0].split(" ")
    rs = []
    for items in ms:
        if items not in model:
            rs.append(unk)
        else:
            rs.append(model['items'])
    vec_jean[i] = np.concatenate((rs[0],rs[1],rs[2]), axis = None)
    model_jean[ls[0]] = vec_jean[i]
    i+=1
    #print(i)

print("done with jean")
#print (vec_jean[1800])
g.close()



def similailr_dict (dress,jean):

    p = open("dress_sim.txt",'w')
    q = open("jean_sim.txt",'w')


    for key in dress:
        sim_dict_dress[key] = []

    for key in jean:
        sim_dict_jean[key] = []
    

    for key, value in dress.items():
    	f =0
    	for k,v in jean.items():
    		cos  = dot(dress[key], jean[k])/(norm(dress[key])*norm(jean[k]))
    		if cos >= 0.8:
    			sim_dict_dress[key].append(k)
    			f = 1
    			p.write(key +"\t"+k+"\n")
    	if f ==1:
    		p.write("*********************\n")
    		f=0

    print ("dress done")


    for key, value in jean.items():
    	f=0
    	for k,v in dress.items():
    		cos  = dot(jean[key], dress[k])/(norm(jean[key])*norm(dress[k]))
    		if cos >= 0.8:
    			sim_dict_jean[key].append(k)
    			f=1
    			q.write(key +"\t"+k+"\n")
    	if f ==1:
    		q.write("*********************\n")
    		f=0


    p.close()
    q.close()
    print ("jean done")





def find_sim_matrix(dress, jean):
	m  = cosine_similarity(dress, jean, dense_output=True)

    



print ("dictionary creation")
similailr_dict(model_dress,model_jean)
#find_sim_matrix(vec_dress, vec_jean )
#print(cosine_similarity([model_dress["this white and"]],[model_jean["Pocket Blue Mid"]]))

