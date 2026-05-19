import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# LIST OF VIDEOS
# ==================================================

video_files = [
    "../videos/Extrait1-Cosmos_Laundromat1(340p).m4v",
    "../videos/Extrait2-ManWithAMovieCamera.m4v",
    "../videos/Extrait3-Vertigo-Dream_Scene(320p).m4v",
    "../videos/Extrait4-Entracte-Poursuite_Corbillard(358p).m4v",
    "../videos/Extrait5-Matrix-Helicopter_Scene(280p).m4v"
]

# ==================================================
# STORE RESULTS FOR ALL VIDEOS
# ==================================================

all_results = []

# ==================================================
# PROCESS EACH VIDEO
# ==================================================

for video_idx, video_file in enumerate(video_files):

    print("\n" + "="*60)
    print(f"Processing video {video_idx+1}: {video_file}")
    print("="*60)

    cap = cv2.VideoCapture(video_file)

    ret, frame1 = cap.read()

    if not ret:
        print("Error opening video")
        continue

    # First frame in grayscale
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    index = 1

    # Lists for statistics
    mean_magnitudes = []
    std_magnitudes = []
    histogram_entropies = []
    frame_indices = []

    # ==================================================
    # READ SECOND FRAME
    # ==================================================

    ret, frame2 = cap.read()

    if not ret:
        cap.release()
        continue

    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # ==================================================
    # MAIN LOOP
    # ==================================================

    while ret:

        index += 1

        # ==============================================
        # DENSE OPTICAL FLOW (Farneback)
        # ==============================================

        flow = cv2.calcOpticalFlowFarneback(
            prvs,
            next,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=7,
            poly_sigma=1.5,
            flags=0
        )

        # ==============================================
        # FLOW COMPONENTS
        # ==============================================

        vx = flow[:,:,0]
        vy = flow[:,:,1]

        # ==============================================
        # MAGNITUDE + ANGLE
        # ==============================================

        mag, ang = cv2.cartToPolar(vx, vy)

        # ==============================================
        # QUESTION 2 :
        # 2D HISTOGRAM OF (Vx,Vy)
        # ==============================================

        maxV = 20

        hist = cv2.calcHist(
            [vx, vy],
            [0,1],
            None,
            [64,64],
            [-maxV,maxV,-maxV,maxV]
        )

        # Normalize histogram as probability distribution
        hist = cv2.normalize(
            hist,
            None,
            alpha=1,
            norm_type=cv2.NORM_L1
        )

        # ==============================================
        # HISTOGRAM ENTROPY
        # ==============================================

        entropy = -np.sum(
            hist * np.log2(hist + 1e-10)
        )

        # ==============================================
        # GLOBAL FLOW STATISTICS
        # ==============================================

        mean_mag = np.mean(mag)
        std_mag = np.std(mag)

        # Store statistics
        mean_magnitudes.append(mean_mag)
        std_magnitudes.append(std_mag)
        histogram_entropies.append(entropy)

        frame_indices.append(index)

        # ==============================================
        # NEXT FRAME
        # ==============================================

        prvs = next

        ret, frame2 = cap.read()

        if ret:
            next = cv2.cvtColor(
                frame2,
                cv2.COLOR_BGR2GRAY
            )

    # ==================================================
    # END VIDEO
    # ==================================================

    cap.release()

    print(f"Total frames processed: {index}")

    # Store all results
    all_results.append({
        'video': video_file,
        'frames': frame_indices,
        'mean_mag': mean_magnitudes,
        'std_mag': std_magnitudes,
        'entropy': histogram_entropies
    })

# ==================================================
# PLOT RESULTS
# ==================================================

fig, axs = plt.subplots(
    len(all_results),
    3,
    figsize=(18, 4*len(all_results)),
    sharex=False
)

# Handle single video case
if len(all_results) == 1:
    axs = [axs]

for i, result in enumerate(all_results):

    frames = result['frames']

    # ==============================================
    # 1. Mean Magnitude
    # ==============================================

    axs[i][0].plot(
        frames,
        result['mean_mag'],
        color='blue',
        linewidth=1.5
    )

    axs[i][0].set_title(
        f'Video {i+1} : Mean Motion Magnitude'
    )

    axs[i][0].set_xlabel('Frame')

    axs[i][0].set_ylabel('Mean |V|')

    axs[i][0].grid(True)

    # ==============================================
    # 2. Standard Deviation
    # ==============================================

    axs[i][1].plot(
        frames,
        result['std_mag'],
        color='orange',
        linewidth=1.5
    )

    axs[i][1].set_title(
        f'Video {i+1} : Motion Dispersion'
    )

    axs[i][1].set_xlabel('Frame')

    axs[i][1].set_ylabel('Std(|V|)')

    axs[i][1].grid(True)

    # ==============================================
    # 3. Histogram Entropy
    # ==============================================

    axs[i][2].plot(
        frames,
        result['entropy'],
        color='green',
        linewidth=1.5
    )

    axs[i][2].set_title(
        f'Video {i+1} : Histogram Entropy'
    )

    axs[i][2].set_xlabel('Frame')

    axs[i][2].set_ylabel('Entropy')

    axs[i][2].grid(True)

# ==================================================
# FINAL DISPLAY
# ==================================================

plt.tight_layout()

plt.show()