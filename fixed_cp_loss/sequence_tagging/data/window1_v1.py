fp = open("tags_in_jean.txt",'r')
gp = open("tags_in_dress.txt",'r')
mp = open("dress_w1.txt",'w')
kp = open("jean_w1.txt",'w')


# Storing top tags of dress into tagdl as per error
tagdl = []
for lines in fp.readlines():
	ks =[]
	ks =lines.strip().split("\n")
	tagdl.append(ks[0])


# Storing all the tags of jeans into tagdl

tagjl = []
for lines in gp.readlines():
	ks =[]
	ks =lines.strip().split("\n")
	tagjl.append(ks[0])



# Writing the dress window
for items in tagdl:
	f = open("new_dress_output.txt",'r')
	for lines in f.readlines():
		ks =[]
		ks= lines.strip().split(" ")
		if len(ks) == 3 and ks[2] == items : 
			mp.write(items + " " + ks[0] + "\n")
	f.close()


# Writing the jean window
for items in tagjl:
	f = open("new_jean_output.txt",'r')
	for lines in f.readlines():
		ks =[]
		ks= lines.strip().split(" ")
		if len(ks) == 3 and ks[2] == items : 
			kp.write(items + " " + ks[0] + "\n")
	f.close()




fp.close()
mp.close()
gp.close()
kp.close()