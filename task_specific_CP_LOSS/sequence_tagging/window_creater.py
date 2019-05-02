f =open('dress_3_45_train.txt','r')
g =open('jean_3_45_train.txt','r')
f1 = open('sentences_dress.txt','w')
g1 = open('sentences_jean.txt','w')


mainl =[]
maint =[]
mainl.append("NULL")
maint.append("O")

for lines in f.readlines():
	ls = lines.strip().split(" ")
	if len(ls) == 2:
		mainl.append(ls[0])
		maint.append(ls[1])

	else:
		mainl.append("NULL")
		maint.append("O")

mainl.append("NULL")
maint.append("O")

for i in range (1,len(mainl)-1):
	if maint[i] == 'O':
		f1.write(mainl[i-1]+" "+mainl[i]+" "+mainl[i+1]+"\t"+maint[i]+"\n")
	else:
		f1.write(mainl[i-1]+" "+mainl[i]+" "+mainl[i+1]+"\t"+maint[i]+"_dress"+"\n")



mainl =[]
maint =[]
mainl.append("NULL")
maint.append("O")

for lines in g.readlines():
	ls = lines.strip().split(" ")
	if len(ls) == 2:
		mainl.append(ls[0])
		maint.append(ls[1])

	else:
		mainl.append("NULL")
		maint.append("O")

mainl.append("NULL")
maint.append("O")

for i in range (1,len(mainl)-1):
	if maint[i] == 'O':
		g1.write(mainl[i-1]+" "+mainl[i]+" "+mainl[i+1]+"\t"+maint[i]+"\n")
	else:
		g1.write(mainl[i-1]+" "+mainl[i]+" "+mainl[i+1]+"\t"+maint[i]+"_jean"+"\n")



f.close()
g.close()
f1.close()
g1.close()