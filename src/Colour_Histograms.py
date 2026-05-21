import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ==================================================
# CREATE OUTPUT DIRECTORY
# ==================================================

os.makedirs('../plots', exist_ok=True)

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
# STORE RESULTS
# ==================================================

all_results = []

# ==================================================
# PROCESS EACH VIDEO
# ==================================================

for video_idx, video_file in enumerate(video_files):

    print("\n" + "="*60)
    print(f"Processing video {video_idx+1}: {video_file}")
    print("="*60)

    video_name = video_file.split('/')[-1].replace('.m4v', '')

    cap = cv2.VideoCapture(video_file)

    ret, frame = cap.read()

    if not ret:
        print("Error opening video")
        continue

    # ==================================================
    # MANUAL MONOCHROME CONFIGURATION
    # ==================================================

    monochrome_videos = [

        "Extrait2-ManWithAMovieCamera",

        "Extrait4-Entracte-Poursuite_Corbillard"
    ]

    is_monochrome = any (
        name in video_name
        for name in monochrome_videos
    )

    # ==================================================
    # INITIAL VARIABLES
    # ==================================================

    prev_hist = None

    chi_distances = []
    bh_distances = []

    frame_indices = []

    index = 0

    # ==================================================
    # MAIN LOOP
    # ==================================================

    while ret:

        # ==============================================
        # HISTOGRAM COMPUTATION
        # ==============================================

        if is_monochrome:

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            hist = cv2.calcHist(
                [gray],
                [0],
                None,
                [64],
                [0, 256]
            )

        else:

            yuv = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2YUV
            )

            hist = cv2.calcHist(
                [yuv],
                [1, 2],
                None,
                [64, 64],
                [0, 256, 0, 256]
            )

        # Normalize histogram

        hist = cv2.normalize(
            hist,
            hist,
            alpha=1,
            norm_type=cv2.NORM_L1
        )

        # ==============================================
        # HISTOGRAM DISTANCES
        # ==============================================

        if prev_hist is not None:

            chi = cv2.compareHist(
                prev_hist,
                hist,
                cv2.HISTCMP_CHISQR
            )

            bh = cv2.compareHist(
                prev_hist,
                hist,
                cv2.HISTCMP_BHATTACHARYYA
            )

            chi_distances.append(chi)
            bh_distances.append(bh)

            frame_indices.append(index)

        prev_hist = hist.copy()

        # ==============================================
        # SAVE REPRESENTATIVE VISUALIZATIONS
        # ==============================================

        if index in [100, 300, 500]:

            fig, axs = plt.subplots(
                1,
                2,
                figsize=(10, 4)
            )

            axs[0].imshow(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            axs[0].set_title(
                f'Frame {index}'
            )

            axs[0].axis('off')

            if is_monochrome:

                axs[1].plot(
                    hist,
                    color='black'
                )

                axs[1].set_title(
                    '1D Luminance Histogram'
                )

            else:

                axs[1].imshow(
                    hist,
                    interpolation='nearest',
                    origin='lower',
                    extent=[0,256,0,256],
                    cmap='jet'
                )

                axs[1].set_title(
                    '2D UV Histogram'
                )

            plt.tight_layout()

            plt.savefig(
                f'../plots/{video_name}_frame_{index}.png',
                dpi=150
            )

            plt.close()

        # ==============================================
        # NEXT FRAME
        # ==============================================

        index += 1

        ret, frame = cap.read()

    # ==================================================
    # END VIDEO
    # ==================================================

    cap.release()

    print(f"Total frames processed: {index}")

    # ==================================================
    # STORE RESULTS
    # ==================================================

    all_results.append({

        'video': video_name,
        'frames': frame_indices,
        'chi': chi_distances,
        'bh': bh_distances
    })

# ==================================================
# PLOT RESULTS
# ==================================================

fig, axs = plt.subplots(
    len(all_results),
    2,
    figsize=(14, 4*len(all_results)),
    sharex=False
)

if len(all_results) == 1:
    axs = [axs]

# ==================================================
# PLOT EACH VIDEO
# ==================================================

for i, result in enumerate(all_results):

    frames = result['frames']

    # ==============================================
    # CHI-SQUARE DISTANCE
    # ==============================================

    axs[i][0].plot(
        frames,
        result['chi'],
        color='blue',
        linewidth=1.5
    )

    axs[i][0].set_title(
        f"{result['video']} : Chi-Square Distance"
    )

    axs[i][0].set_xlabel('Frame')

    axs[i][0].set_ylabel('Chi-Square')

    axs[i][0].grid(True)

    # ==============================================
    # BHATTACHARYYA DISTANCE
    # ==============================================

    axs[i][1].plot(
        frames,
        result['bh'],
        color='red',
        linewidth=1.5
    )

    axs[i][1].set_title(
        f"{result['video']} : Bhattacharyya Distance"
    )

    axs[i][1].set_xlabel('Frame')

    axs[i][1].set_ylabel('Bhattacharyya')

    axs[i][1].grid(True)

# ==================================================
# FINAL DISPLAY
# ==================================================

plt.tight_layout()

plt.savefig(
    '../plots/q1_histogram_analysis.png',
    dpi=150
)

print("\nPlots saved successfully.")