import cv2
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
import matplotlib.pyplot as plt

# List of video files to process
video_files = [
    "../videos/Extrait1-Cosmos_Laundromat1(340p).m4v", 
    "../videos/Extrait2-ManWithAMovieCamera.m4v",
    "../videos/Extrait3-Vertigo-Dream_Scene(320p).m4v",
    "../videos/Extrait4-Entracte-Poursuite_Corbillard(358p).m4v",
    "../videos/Extrait5-Matrix-Helicopter_Scene(280p).m4v"  
]

# Process each video file
all_results = []

for video_idx, video_file in enumerate(video_files):
    print(f"\n{'='*60}")
    print(f"Processing video {video_idx + 1}/5: {video_file}")
    print(f"{'='*60}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_file)
    
    # Read the first frame to check if the video stream is opened successfully
    ret, frame = cap.read()
    if not ret:
        print(f"Error opening video stream: {video_file}")
        continue
    
    # Initialize variables to store the previous histogram and the distances between histograms
    prev_hist = None
    distances_bh = []
    distances_chi = []
    distances_inter = []
    distances_correl = []
    frame_indices = []
    index = 0
    
    # Process frames from current video
    while ret:
        
        # Check if the frame is monochrome (grayscale)
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            # For monochrome videos, create a dummy color frame
            if index == 0:
                print(f"  [INFO] Monochrome video detected")
            frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            yuv = cv2.cvtColor(frame_color, cv2.COLOR_BGR2YUV)
        else:
            # Convert the frame to YUV color space for color videos
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        
        # QUESTION 1: Calculate the 2D histogram for the U(1) and V(2) channels
        # This histogram represents the joint probability of chromatic components (u, v)
        hist = cv2.calcHist(
            [yuv],
            [1, 2],        # U and V channels
            None,
            [64, 64],      # bins (64x64 for 2D histogram)
            [0, 256, 0, 256]
        )
        
        # Normalize the histogram 
        hist_norm = cv2.normalize(
            hist,
            None,
            0,
            255,
            cv2.NORM_MINMAX)
        
        hist_norm = hist_norm.astype(np.uint8)
        
        # Calculate the distance between the current histogram and the previous one
        if prev_hist is not None:
            # Use the Chi-Square distance to compare the histograms
            dist_chi = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
            distances_chi.append(dist_chi)
            # Use the Bhattacharyya distance to compare the histograms
            dist_bh = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            distances_bh.append(dist_bh)
            # Use the Intersection distance to compare the histograms
            dist_inter = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_INTERSECT)
            distances_inter.append(dist_inter)
            # Use the Correlation distance to compare the histograms
            dist_correl = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            distances_correl.append(dist_correl)
            # Store the frame index for plotting
            frame_indices.append(index)
        
        # Update the previous histogram
        prev_hist = hist.copy()
        
        index += 1  
        
        # Display progress
        if index % 100 == 0:
            print(f"  Processed {index} frames...")
        
        # Read the next frame
        ret, frame = cap.read()
    
    # Release video capture
    cap.release()
    
    print(f"  Total frames processed: {index}")
    
    # SCENE CHANGE DETECTION based on histogram distances
    print("\n  === Scene Change Detection ===")
    scene_cuts = []
    
    if len(distances_chi) > 0:
        # Calculate threshold as a multiple of the standard deviation
        mean_chi = np.mean(distances_chi)
        std_chi = np.std(distances_chi)
        threshold_chi = mean_chi + 2.5 * std_chi
        
        # Detect cuts where Chi-Square distance exceeds threshold
        for i, dist in enumerate(distances_chi):
            if dist > threshold_chi:
                scene_cuts.append(i + 1)  # +1 because distances start from frame 1
                print(f"    Frame {i + 1}: Chi-Square={dist:.2f} (threshold: {threshold_chi:.2f})")
        
        print(f"\n  Statistics (Chi-Square):")
        print(f"    Mean distance: {mean_chi:.4f}")
        print(f"    Std deviation: {std_chi:.4f}")
        print(f"    Threshold: {threshold_chi:.4f}")
        print(f"    Total scene cuts detected: {len(scene_cuts)}")
        
        # Create individual plot for this video
        fig_single, axs_single = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # 1. Chi-Square
        axs_single[0].plot(frame_indices, distances_chi, color='b', label=r'Chi-Square ($\chi^2$)', linewidth=2)
        axs_single[0].axhline(y=threshold_chi, color='r', linestyle='--', label=f'Threshold: {threshold_chi:.2f}')
        axs_single[0].set_ylabel('Absolute Value')
        axs_single[0].set_title(f'Video {video_idx+1}: Chi-Square Distance (Peaks UP on cuts)')
        axs_single[0].grid(True)
        axs_single[0].legend(loc='upper right')
        
        # 2. Bhattacharyya
        axs_single[1].plot(frame_indices, distances_bh, color='orange', label='Bhattacharyya', linewidth=2)
        axs_single[1].set_ylabel('Value [0, 1]')
        axs_single[1].set_title(f'Video {video_idx+1}: Bhattacharyya Distance (Peaks UP on cuts)')
        axs_single[1].grid(True)
        axs_single[1].legend(loc='upper right')
        
        # 3. Intersection
        axs_single[2].plot(frame_indices, distances_inter, color='g', label='Intersection', linewidth=2)
        axs_single[2].set_ylabel('Absolute Value')
        axs_single[2].set_title(f'Video {video_idx+1}: Histogram Intersection (Drops DOWN on cuts)')
        axs_single[2].grid(True)
        axs_single[2].legend(loc='upper right')
        
        # 4. Correlation
        axs_single[3].plot(frame_indices, distances_correl, color='r', label='Correlation', linewidth=2)
        axs_single[3].set_ylabel('Value [-1, 1]')
        axs_single[3].set_xlabel('Frame Index')
        axs_single[3].set_title(f'Video {video_idx+1}: Correlation Coefficient (Drops DOWN on cuts)')
        axs_single[3].grid(True)
        axs_single[3].legend(loc='upper right')
        
        plt.tight_layout()
        # Save individual plot
        video_name = video_file.split('/')[-1].replace('.m4v', '')
        plot_filename = f'../plots/histogram_video_{video_idx+1}_{video_name}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to: {plot_filename}")
        plt.close(fig_single)
        
        # Store results for this video
        all_results.append({
            'video': video_file,
            'frames': index,
            'scene_cuts': len(scene_cuts),
            'distances_chi': distances_chi,
            'distances_bh': distances_bh,
            'distances_inter': distances_inter,
            'distances_correl': distances_correl,
            'frame_indices': frame_indices
        })

# Create comprehensive plots for all videos
print(f"\n{'='*60}")
print("Creating comparison plots...")
print(f"{'='*60}")

if all_results:
    # Create a figure with subplots for each video
    fig, axs = plt.subplots(len(all_results), 4, figsize=(16, 4*len(all_results)))
    
    # Handle single video case (no nested array)
    if len(all_results) == 1:
        axs = [axs]
    
    for video_idx, result in enumerate(all_results):
        distances_chi = result['distances_chi']
        distances_bh = result['distances_bh']
        distances_inter = result['distances_inter']
        distances_correl = result['distances_correl']
        frame_indices = result['frame_indices']
        
        # Calculate threshold
        mean_chi = np.mean(distances_chi) if distances_chi else 0
        std_chi = np.std(distances_chi) if distances_chi else 0
        threshold_chi = mean_chi + 2.5 * std_chi
        
        # 1. Chi-Square
        axs[video_idx][0].plot(frame_indices, distances_chi, color='b', linewidth=1)
        axs[video_idx][0].axhline(y=threshold_chi, color='r', linestyle='--', alpha=0.7)
        axs[video_idx][0].set_ylabel('Chi-Square')
        axs[video_idx][0].set_title(f"Video {video_idx+1}: Chi-Square Distance")
        axs[video_idx][0].grid(True, alpha=0.3)
        
        # 2. Bhattacharyya
        axs[video_idx][1].plot(frame_indices, distances_bh, color='orange', linewidth=1)
        axs[video_idx][1].set_ylabel('Bhattacharyya')
        axs[video_idx][1].set_title(f"Video {video_idx+1}: Bhattacharyya Distance")
        axs[video_idx][1].grid(True, alpha=0.3)
        
        # 3. Intersection
        axs[video_idx][2].plot(frame_indices, distances_inter, color='g', linewidth=1)
        axs[video_idx][2].set_ylabel('Intersection')
        axs[video_idx][2].set_title(f"Video {video_idx+1}: Histogram Intersection")
        axs[video_idx][2].grid(True, alpha=0.3)
        
        # 4. Correlation
        axs[video_idx][3].plot(frame_indices, distances_correl, color='r', linewidth=1)
        axs[video_idx][3].set_ylabel('Correlation')
        axs[video_idx][3].set_title(f"Video {video_idx+1}: Correlation Coefficient")
        axs[video_idx][3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../plots/histogram_distances_all_videos.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: ../plots/histogram_distances_all_videos.png")


fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# Calculate threshold lines for scene cut detection
mean_chi = np.mean(distances_chi) if distances_chi else 0
std_chi = np.std(distances_chi) if distances_chi else 0
threshold_chi = mean_chi + 2.5 * std_chi

# 1. Chi-Square
axs[0].plot(frame_indices, distances_chi, color='b', label=r'Chi-Square ($\chi^2$)', linewidth=2)
axs[0].axhline(y=threshold_chi, color='r', linestyle='--', label=f'Threshold: {threshold_chi:.2f}')
axs[0].set_ylabel('Absolute Value')
axs[0].set_title('Chi-Square Distance (Peaks UP on cuts)')
axs[0].grid(True)
axs[0].legend(loc='upper right')

# 2. Bhattacharyya
axs[1].plot(frame_indices, distances_bh, color='orange', label='Bhattacharyya', linewidth=2)
axs[1].set_ylabel('Value [0, 1]')
axs[1].set_title('Bhattacharyya Distance (Peaks UP on cuts) ')
axs[1].grid(True)
axs[1].legend(loc='upper right')

# 3. Intersection
axs[2].plot(frame_indices, distances_inter, color='g', label='Intersection', linewidth=2)
axs[2].set_ylabel('Absolute Value')
axs[2].set_title('Histogram Intersection (Drops DOWN on cuts) ')
axs[2].grid(True)
axs[2].legend(loc='upper right')

# 4. Correlation
axs[3].plot(frame_indices, distances_correl, color='r', label='Correlation', linewidth=2)
axs[3].set_ylabel('Value [-1, 1]')
axs[3].set_xlabel('Frame Index')
axs[3].set_title('Correlation Coefficient (Drops DOWN on cuts)')
axs[3].grid(True)
axs[3].legend(loc='upper right')


plt.tight_layout()
# Save the plot to file instead of displaying (for headless operation)
plt.savefig('../plots/histogram_distances_analysis.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to: ../plots/histogram_distances_analysis.png")
# plt.show()  # Commented out for headless operation