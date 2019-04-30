import numpy as np
from numpy import dot
from numpy.linalg import norm




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



def similarity_list_generator(model, dress ,jean):
	unk = np.zeros(300, dtype=np.float32)
	f = open("dress_list.txt",'w')
	g = open("jean_list.txt",'w')
	#print (len(dress))
	#print (len(jean))
	for d in dress:
		for j in jean:
			if d[0] not in model:
				continue
			else:
				a = model[d[0]]
				#print (d[0])

			if j[0] not in model:
				continue
			else:
				b = model[j[0]]
				#print (j[0])

			if (norm(a)*norm(b)) == 0:
				continue

			cos  = dot(a, b)/(norm(a)*norm(b))
			
			if (cos >= 0.5):
				f.write(d[0]+" "+d[1]+"_dress"+" "+ j[0] + " "+ j[1]+"_jean"+"\t"+str(cos)+"\n")
				g.write(j[0]+" "+j[1]+"_jean"+" "+ d[0] + " "+ d[1]+"_dress"+"\t"+str(cos)+"\n")
	f.close()
	g.close()


def lambda_calculator():

	tot={}
	eight={}
	five={}
	with open('dress_list.txt') as f:
		for line in f:
			data=line.rstrip().split('\t')
			key=data[0]
			fval=float(data[1])
			if data[0] in tot:
				tot[data[0]]+=1
			else:
				tot[data[0]]=1

			if fval > 0.8:
				if key in eight:
					eight[key]+=1
				else:
					eight[key]=1
			if fval > 0.5:
				if key in five:
					five[key]+=1
				else:
					five[key]=1

	g = open('dress_jean_lambda.txt','w')

	for key in tot:
		if key not in eight:
			eight[key]=0
		if key not in five:
			five[key]=0
		#print (key,tot[key],eight[key],five[key],eight[key]/five[key])
		g.write(key+"\t"+str(tot[key])+"\t"+str(eight[key]/five[key])+"\n")
	g.close()



def main():
	file = "glove.6B.300d.txt"
	model= loadGloveModel(file) 
	
	dress_list = []
	jean_list =[]

	#taking items from dress
	f = open("dress_3_45_train.txt",'r')
	for lines in f.readlines():
		ls = lines.strip().split(" ")
		if len(ls) == 2 and ls[1] != 'O':
			dress_list.append(ls)
	f.close()


	#taking items from jean
	g = open("jean_3_45_train.txt",'r')
	for lines in g.readlines():
		ls = lines.strip().split(" ")
		if len(ls) == 2 and ls[1] != 'O':
			jean_list.append(ls)
	g.close()

	print ("generating similarity list")
	similarity_list_generator(model, dress_list,jean_list)

	print ("generating final lambda similarity list")
	lambda_calculator()






  
if __name__== "__main__":
  main()






