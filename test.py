n = int(input())
store = [int(x) for x in input().split()]
count = 0
dict = {}
c = 0
for i in range(n-1):
	c = 0
	for j in range(i+1,n):
		if store[i]>store[j]:
			c += 1
			count += 1
	dict[store[i]]=c
Max = 0
K = 0
for key in dict.keys():
	ind = store.index(key)
	sub = dict[key]-ind
	if Max <=sub:
		K = ind
		Max = sub
print(count-Max,K+1)