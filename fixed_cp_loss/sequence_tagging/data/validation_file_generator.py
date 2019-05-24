f = open("dress_3_45_test.txt",'r')
g = open("jean_3_45_test.txt",'r')
fg = open("dress_jean_val.txt",'w')

#flag = 0
for lines in f.readlines():
	ls = lines.strip().split(" ")
	if len(ls) == 2:
		if ls[1] == 'O':
			fg.write(lines.strip() + " 1\n")
		else:
			fg.write(ls[0]+" "+ls[1]+"_dress"+" 1\n")

	else:

		fg.write("\n")

		while(1):
			l = g.readline()
			if (l == ''):
				#flag =1
				break
			gs = l.strip().split(" ")
			if (len(gs) != 2):
				fg.write("\n")
				break

			else:
				if gs[1] == 'O':
					fg.write(l.strip() + " 2\n")
				else:
					fg.write(gs[0]+" "+gs[1]+"_jean"+" 2\n")

	#if flag ==1:
		#break



f.close()
g.close()
fg.close()
