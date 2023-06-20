import cv2
import numpy as np

# Step 1: Create a random number of dots in a random color on the first image.
img1 = np.zeros((512, 512, 3), np.uint8)  # create a blank image
num_dots = np.random.randint(10, 20)  # generate a random number of dots
dots1 = np.empty([])
for i in range(num_dots):
    color = tuple(np.random.randint(0, 255, size=3).tolist())  # generate a random color
    pt = (np.random.randint(0, 512), np.random.randint(0, 512))  # generate a random point
    if np.ndim(dots1) == 0: dots1= (pt[0],pt[1],color)
    else:dots1=np.vstack((dots1,(pt[0],pt[1],color)))
    cv2.circle(img1, pt, 5, color, -1)  # draw a circle at the point with the color


# Step 2: Create a Voronoi diagram from the dots in the first image and paint each area with the color corresponding to the dot.
rect = (0, 0, 512, 512)
subdiv = cv2.Subdiv2D(rect)

for dot1 in dots1:
    subdiv.insert((int(dot1[0]), int(dot1[1])))

facets, centers = subdiv.getVoronoiFacetList([])

img_voro = img1.copy()

for i, dot1 in enumerate(f.astype(int) for f in facets):
    cv2.fillPoly(img_voro, [dot1], dots1[i][2])


# Step 3: Place a random number of dots on the second image. At this time, the color of the dots should be the same as the Voronoi diagram.
img2 = np.zeros((512, 512, 3), np.uint8)  # create a blank image
num_dots = np.random.randint(10, 20)  # generate a random number of dots
for i in range(num_dots):
    pt = (np.random.randint(0, 512), np.random.randint(0, 512))  # generate a random point
    color = tuple(img_voro[pt[1], pt[0]].tolist())  # get the color at the point from the Voronoi diagram
    cv2.circle(img2, pt, 5, color, -1)  # draw a circle at the point with the color

# Step 4: Display the two images and the Voronoi diagram.
cv2.imshow('Image 1', img1)
cv2.imshow('Voronoi Diagram', img_voro)
cv2.imshow('Image 2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
