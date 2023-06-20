import random

connected = []
for _ in range(10):
    d1, d2 = sorted(random.sample(range(1, 11), 2))
    if not any(d1 in connected_pair for connected_pair in connected):
        if not any(d2 in connected_pair for connected_pair in connected):
            connected.append([d1, d2])
print(connected)