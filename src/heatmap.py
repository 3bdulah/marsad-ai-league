import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_fake_positions(num_points=200):
    return np.random.randint(0, 640, (num_points, 2))  # x, y positions in 640x640 space

def generate_heatmap(positions, image_shape=(640, 640)):
    heatmap = np.zeros(image_shape, dtype=np.float32)

    # Add "heat" for each point
    for x, y in positions:
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            heatmap[y, x] += 1

    # Smoothen the heatmap
    heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)
    return heatmap

def show_heatmap(heatmap):
    plt.imshow(heatmap, cmap='hot')
    plt.title("📊 Crowd / Player Heatmap")
    plt.colorbar()
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    print("🔥 Generating fake heatmap...")
    positions = generate_fake_positions()
    heatmap = generate_heatmap(positions)
    show_heatmap(heatmap)