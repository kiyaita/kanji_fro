import cv2
import numpy as np

def voronoi_images(img1, dots1, rand_min,rand_max):
    
    # Step 2: Create a Voronoi diagram from the dots in the first image and paint each area with the color corresponding to the dot.
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    for dot1 in dots1:
        subdiv.insert((int(dot1[0]), int(dot1[1])))

    facets, centers = subdiv.getVoronoiFacetList([])

    img_voro = img1.copy()

    for i, dot1 in enumerate(f.astype(int) for f in facets):
        cv2.fillPoly(img_voro, [dot1], dots1[i][2])

    dots2 = np.empty([])
    # Step 3: Place a random number of dots on the second image. At this time, the color of the dots should be the same as the Voronoi diagram.
    img2 = np.zeros((h, w, 3), np.uint8)  # create a blank image
    num_dots = np.random.randint(rand_min, rand_max)  # generate a random number of dots
    for i in range(num_dots):
        pt = (np.random.randint(0, w), np.random.randint(0, h))  # generate a random point
        color = tuple(img_voro[pt[1], pt[0]].tolist())  # get the color at the point from the Voronoi diagram
        if np.ndim(dots2) == 0: dots2= [[pt[0],pt[1],color]]
        else:dots2=np.vstack((dots2,(pt[0],pt[1],color)))
        cv2.circle(img2, pt, 5, color, -1)  # draw a circle at the point with the color
    return img2, dots2, img_voro

np.random.seed(307)
w,h = 1024, 512
rand_min, rand_max = 300, 400
img = np.zeros((h, w, 3), np.uint8)  # create a blank image

# Step 1: Create a random number of dots in a random color on the first image.
num_dots = np.random.randint(rand_min, rand_max)  # generate a random number of dots
dots1 = np.empty([])
for i in range(num_dots):
    color = tuple(np.random.randint(0, 255, size=3).tolist())  # generate a random color
    pt = (np.random.randint(0, w), np.random.randint(0, h))  # generate a random point
    if np.ndim(dots1) == 0: dots1= [[pt[0],pt[1],color]]
    else:dots1=np.vstack((dots1,(pt[0],pt[1],color)))
    cv2.circle(img, pt, 5, color, -1)  # draw a circle at the point with the color


# Set up the output video file
out_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (img.shape[1], img.shape[0]))



for i in range(500):
    # Apply the function to the current image
    img, dots1, img_voro = voronoi_images(img, dots1, rand_min, rand_max)
    # Write the current image to the video file
    out_video.write(img_voro)


out_video.release()

# Step 4: Display the two images and the Voronoi diagram.
"""cv2.imshow('Voronoi Diagram', img_voro)
cv2.waitKey(0)
cv2.destroyAllWindows()"""
