from random import shuffle

f = open("dress_3_45_test.txt",'r')
g = open("dress_3_45_train.txt",'r')
f1 = open("dress_3_45_test1.txt",'w')
g1 = open("dress_3_45_train1.txt",'w')

total_list = []
temp = []
for lines in f.readlines():
	ls = lines.strip().split(" ")
	if len(ls) == 2:
		temp.append(lines)
	else:
		total_list.append(temp)
		temp= []


a = len(total_list)
print a

temp =[]
for lines in g.readlines():
	ls = lines.strip().split(" ")
	if len(ls) == 2:
		temp.append(lines)
	else:
		total_list.append(temp)
		temp= []

b =  len(total_list) 

print str(a) + " " + str(b-a) + " " + str(b)


shuffle(total_list)

train_size  = 0.7 * b

for i in range(0,b):
	if i <= train_size:
		for items in total_list[i]:
			g1.write(items)
		g1.write("\n")

	else:
		for items in total_list[i]:
			f1.write(items)
		f1.write("\n")




f.close()
g.close()
f1.close()
g1.close()