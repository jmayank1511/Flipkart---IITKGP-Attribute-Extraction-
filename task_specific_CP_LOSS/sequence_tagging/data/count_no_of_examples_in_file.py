import sys
filename = sys.argv[1]
f = open(filename ,"r")
data = f.read()
print(1+data.count("\n \n"))
