import cv2
import numpy as np

import pprint

w = h = 360

n = 5
np.random.seed(0)
pts = np.random.randint(0, w, (n, 2))

# [[172  47]
#  [117 192]
#  [323 251]
#  [195 359]
#  [  9 211]
#  [277 242]]



img = np.zeros((w, h, 3), np.uint8)
###########
for p in pts:
    cv2.drawMarker(img, tuple(p), (255, 255, 255), thickness=2)

rect = (0, 0, w, h)

subdiv = cv2.Subdiv2D(rect)

for p in pts:
    subdiv.insert((int(p[0]), int(p[1])))
    
facets, centers = subdiv.getVoronoiFacetList([])

img_draw = img.copy()

cv2.polylines(img_draw, [f.astype(int) for f in facets], True, (255, 255, 255), thickness=2)

step = int(255 / len(facets))

for i, p in enumerate(f.astype(int) for f in facets):
    cv2.fillPoly(img_draw, [p], tuple(np.random.randint(0, 255, size=3).tolist()))
print(img_draw)
cv2.imshow('Voronoi 1', img_draw)
cv2.waitKey(0)