import cv2
import numpy as np
import random

# Define parameters
n_dots = 70        # Number of dots
dot_size = 2      # Size of dots
max_speed = 1      # Maximum speed of dots
connect_distance = 70  # Distance for connecting dots
connect_distance_delta = 15
# Create black image
img_size = (240, 360, 3)
img = np.zeros(img_size, np.uint8)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("workspace7//video.mp4", fourcc, 30,(img_size[1],img_size[0]))



# Generate random dots
dots = []
for i in range(n_dots):
    x = random.randint(0, img_size[1])
    y = random.randint(0, img_size[0])
    dx = random.uniform(-max_speed, max_speed)
    dy = random.uniform(-max_speed, max_speed)
    dots.append((x, y, dx, dy))

# Main loop
for _ in range (600):
    # Clear image
    img.fill(0)
    connect_distance = connect_distance + random.uniform(-connect_distance_delta, connect_distance_delta-1)
    if connect_distance < 0:connect_distance=0
    print(connect_distance)
    # Update dots position
    for i, (x, y, dx, dy) in enumerate(dots):
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
        dots[i] = (x, y, dx, dy)

        # Draw dot
        cv2.circle(img, (int(x), int(y)), dot_size, (255, 255, 255), -1)

    # Connect dots
    connected = []
    for i, dot1 in enumerate(dots):
        for j, dot2 in enumerate(dots):
            if i == j:
                continue
            if dot2 in connected:
                continue
            dist = np.sqrt((dot1[0] - dot2[0])**2 + (dot1[1] - dot2[1])**2)
            if dist < connect_distance:
                cv2.line(img, (int(dot1[0]), int(dot1[1])), (int(dot2[0]), int(dot2[1])), (255, 255, 255), 1)
                connected.append(dot1)
                connected.append(dot2)
                break

    # Show image
    out.write(img)
    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break
out.release()
cv2.destroyAllWindows()
