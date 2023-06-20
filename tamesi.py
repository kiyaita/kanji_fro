x = set([1])
for i in x:
    x.add(x.pop()+1)
print(x)