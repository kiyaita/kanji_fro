import cv2
import numpy as np
import random

# Define parameters
n_dots = 50        # Number of dots
dot_size = 3       # Size of dots
max_speed = 1      # Maximum speed of dots
connect_distance = 70  # Distance for connecting dots
separate_distance = 300


# Create black image
img_size = (500, 500, 3)
img = np.zeros(img_size, np.uint8)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("workspace7//video2.mp4", fourcc, 40,(img_size[1],img_size[0]))

# Generate random dots
dots = []
for i in range(n_dots):
    x = random.randint(0, img_size[1])
    y = random.randint(0, img_size[0])
    dx = random.uniform(-max_speed, max_speed)
    dy = random.uniform(-max_speed, max_speed)
    id = i
    dots.append((id, x, y, dx, dy))

connected = []#初期化:id
# Main loop
while True:
    # Clear image
    img.fill(0)
    # Update dots position
    for i,(id, x, y, dx, dy) in enumerate(dots):
        # Update position
        x += dx
        y += dy

        # Bounce off edges
        if x < 0 or x > img_size[1]:
            dx = -dx
            x += dx
        if y < 0 or y > img_size[0]:
            dy = -dy
            y += dy

        # Update dot
        dots[i] = (id, x, y, dx, dy)

        # Draw dot
        cv2.circle(img, (int(x), int(y)), dot_size, (255, 255, 255), -1)

    # Connect dots
    for j, (dotA, dotB) in enumerate(connected):
        print(dots[dotA][1],dots[dotA][2])
        if np.sqrt((dots[dotA][1]-dots[dotB][1])**2 + (dots[dotA][2]-dots[dotB][2])**2) > separate_distance:
            connected.pop(j)
    
    for i, dot1 in enumerate(dots):
        if any(dot1[0] in connected_pair for connected_pair in connected):
            continue
        closest_dist = connect_distance
        closest_dot = None
        for j, dot2 in enumerate(dots):
            if i == j:
                continue
            if any(dot2[0] in connected_pair for connected_pair in connected):
                continue
            
            
            dist = np.sqrt((dot1[1] - dot2[1])**2 + (dot1[2] - dot2[2])**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_dot = dot2
                
        if closest_dot is not None:
            connected.append([dot1[0],closest_dot[0]])
            
    for j, (dotA, dotB) in enumerate(connected):
        cv2.line(img, (int(dots[dotA][1]), int(dots[dotA][2])), (int(dots[dotB][1]), int(dots[dotB][2])), (255, 255, 255), 1)
            
            
    print(connected)
    # Show image
    out.write(img)
    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break
out.release()
cv2.destroyAllWindows()
