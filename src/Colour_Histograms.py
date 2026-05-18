import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

# Open the video file 
# cap = cv2.VideoCapture("../videos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")
cap = cv2.VideoCapture("../videos/Extrait2-ManWithAMovieCamera.m4v")


# Read the first frame to check if the video stream is opened successfully
ret, frame = cap.read() 
if not ret:
    print("Error opening video stream")
    exit()

# Initialize variables to store the previous histogram and the distances between histograms
prev_hist = None
distances_bh = []
distances_chi = []
distances_inter = []
distances_correl = []
frame_indices = []
index = 0

while(ret):
    
    # Convert the frame to YUV color space
    yuv = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV)
    
    # Calculate the 2D histogram for the U(1) and V(2) channels
    # hist = cv2.calcHist([yuv], [1,2], None, [256,256], [0,255,0,255])
    # Calculate the 1D histogram for the Y(0) channel (luminance)
    hist= cv2.calcHist([yuv], [0], None, [256], [0,255])
    
    # Normalize the histogram 
    hist_norm = cv2.GaussianBlur(hist,(5,5),cv2.BORDER_DEFAULT)
    hist_norm = ((hist_norm*255.0)/np.amax(hist_norm)).astype(np.uint8)
    
    # Calculate the distance between the current histogram and the previous one
    if prev_hist is not None:
        # Use the Chi-Square distance to compare the histograms (lower values indicate more similarity between the histograms)
        dist_chi = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
        distances_chi.append(dist_chi)
        # Use the Bhattacharyya distance to compare the histograms (lower values indicate more similarity between the histograms)
        dist_bh = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
        distances_bh.append(dist_bh)
        # Use the Intersection distance to compare the histograms (higher values indicate more similarity between the histograms)
        dist_inter = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_INTERSECT)
        distances_inter.append(dist_inter)
        # Use the Correlation distance to compare the histograms (higher values indicate more similarity between the histograms)
        dist_correl = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
        distances_correl.append(dist_correl)
        # Store the frame index for plotting
        frame_indices.append(index)

    # Update the previous histogram
    prev_hist = hist.copy()

    index += 1  

    # Display the frame and the histogram
    cv2.imshow('Image',frame)
    hist_display = cv2.applyColorMap(hist_norm,cv2.COLORMAP_JET)
    cv2.imshow('Histogramme (u,v)',hist_display)
    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame_%04d.png'%index,frame)
        cv2.imwrite('Hist_UV_%04d.png'%index,hist_display)
    
    # Read the next frame
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()

# Plot the distances between histograms for each method
fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# 1. Chi-Square
axs[0].plot(frame_indices, distances_chi, color='b', label='Chi-Square ($\chi^2$)')
axs[0].set_ylabel('Absolute Value')
axs[0].set_title('Chi-Square Distance (Pikes UP on cuts)')
axs[0].grid(True)
axs[0].legend(loc='upper right')

# 2. Bhattacharyya
axs[1].plot(frame_indices, distances_bh, color='orange', label='Bhattacharyya')
axs[1].set_ylabel('Value [0, 1]')
axs[1].set_title('Bhattacharyya Distance (Pikes UP on cuts) ')
axs[1].grid(True)
axs[1].legend(loc='upper right')

# 3. Intersection
axs[2].plot(frame_indices, distances_inter, color='g', label='Intersection')
axs[2].set_ylabel('Absolute Value')
axs[2].set_title('Histogram Intersection (Drops DOWN on cuts) ')
axs[2].grid(True)
axs[2].legend(loc='upper right')

# 4. Correlation
axs[3].plot(frame_indices, distances_correl, color='r', label='Correlation')
axs[3].set_ylabel('Value [-1, 1]')
axs[3].set_xlabel('Frame Index')
axs[3].set_title('Correlation Coefficient (Drops DOWN on cuts)')
axs[3].grid(True)
axs[3].legend(loc='upper right')


plt.tight_layout()
plt.show()