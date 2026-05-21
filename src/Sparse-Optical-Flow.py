import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==================================================
# CREATE OUTPUT DIRECTORY
# ==================================================

os.makedirs("../plots", exist_ok=True)

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
# MANUAL MONOCHROME CONFIGURATION
# ==================================================

monochrome_videos = [
    "Extrait2-ManWithAMovieCamera",
    "Extrait4-Entracte-Poursuite_Corbillard"
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

    # ==================================================
    # MANUAL MONOCHROME DETECTION
    # ==================================================

    video_name = (
        video_file
        .split('/')[-1]
        .replace('.m4v', '')
    )

    is_monochrome = any(
        name in video_name
        for name in monochrome_videos
    )

    if is_monochrome:
        print("[INFO] Monochrome video detected")
    else:
        print("[INFO] Color video detected")

    # ==================================================
    # INITIAL FRAME PREPARATION
    # ==================================================

    previous_gray = cv2.cvtColor(
        frame1,
        cv2.COLOR_BGR2GRAY
    )

    h, w = previous_gray.shape[:2]

    index = 1

    # ==================================================
    # STORE STATISTICS
    # ==================================================

    mean_residuals = []
    residual_entropies = []
    histogram_differences = []
    frame_indices = []

    detected_cuts = []

    last_cut_frame = -100

    previous_hist_difference = 0

    # ==================================================
    # INITIAL HISTOGRAM
    # ==================================================

    if is_monochrome:

        previous_hist = cv2.calcHist(
            [previous_gray],
            [0],
            None,
            [128],
            [0, 256]
        )

    else:

        previous_hist = cv2.calcHist(
            [frame1],
            [0, 1, 2],
            None,
            [8, 8, 8],
            [0, 256, 0, 256, 0, 256]
        )

    previous_hist = cv2.normalize(
        previous_hist,
        previous_hist
    )

    # ==================================================
    # READ SECOND FRAME
    # ==================================================

    ret, frame2 = cap.read()

    if not ret:
        cap.release()
        continue

    current_gray = cv2.cvtColor(
        frame2,
        cv2.COLOR_BGR2GRAY
    )

    # ==================================================
    # MAIN LOOP
    # ==================================================

    while ret:

        index += 1

        # ==============================================
        # DENSE OPTICAL FLOW (Farneback)
        # ==============================================

        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
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

        vx = flow[:, :, 0]
        vy = flow[:, :, 1]

        # ==============================================
        # PREDICTED FRAME GENERATION
        # ==============================================

        grid_x, grid_y = np.meshgrid(
            np.arange(w),
            np.arange(h)
        )

        map_x = (grid_x + vx).astype(np.float32)
        map_y = (grid_y + vy).astype(np.float32)

        predicted_frame = cv2.remap(
            previous_gray,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        # ==============================================
        # RESIDUAL ERROR IMAGE
        # ==============================================

        residual_image = cv2.absdiff(
            current_gray,
            predicted_frame
        )

        # ==============================================
        # RESIDUAL HISTOGRAM
        # ==============================================

        hist_residual = cv2.calcHist(
            [residual_image],
            [0],
            None,
            [256],
            [0, 256]
        )

        hist_residual = cv2.normalize(
            hist_residual,
            None,
            alpha=1,
            norm_type=cv2.NORM_L1
        )

        # ==============================================
        # RESIDUAL ENTROPY
        # ==============================================

        entropy_residual = -np.sum(
            hist_residual *
            np.log2(hist_residual + 1e-10)
        )

        # ==============================================
        # MEAN RESIDUAL ERROR
        # ==============================================

        mean_residual = np.mean(
            residual_image
        )

        # ==============================================
        # CURRENT HISTOGRAM
        # ==============================================

        if is_monochrome:

            current_hist = cv2.calcHist(
                [current_gray],
                [0],
                None,
                [128],
                [0, 256]
            )

        else:

            current_hist = cv2.calcHist(
                [frame2],
                [0, 1, 2],
                None,
                [8, 8, 8],
                [0, 256, 0, 256, 0, 256]
            )

        current_hist = cv2.normalize(
            current_hist,
            current_hist
        )

        # ==============================================
        # HISTOGRAM DIFFERENCE
        # ==============================================

        hist_difference = cv2.compareHist(
            previous_hist,
            current_hist,
            cv2.HISTCMP_BHATTACHARYYA
        )

        # ==============================================
        # STORE STATISTICS
        # ==============================================

        mean_residuals.append(mean_residual)

        residual_entropies.append(
            entropy_residual
        )

        histogram_differences.append(
            hist_difference
        )

        frame_indices.append(index)

        # ==============================================
        # ADAPTIVE THRESHOLDS
        # ==============================================

        if len(residual_entropies) > 20:

            local_entropy = residual_entropies[-20:]

            entropy_mean = np.mean(local_entropy)

            entropy_std = np.std(local_entropy)

            entropy_threshold = (
                entropy_mean +
                2.0 * entropy_std
            )

            residual_threshold = (
                np.mean(mean_residuals[-20:]) +
                2.0 * np.std(mean_residuals[-20:])
            )

            # ==========================================
            # CUT CONDITIONS
            # ==========================================

            strong_residual = (
                entropy_residual >
                entropy_threshold
            )

            strong_hist_change = (
                hist_difference > 0.38
            )

            strong_mean_residual = (
                mean_residual >
                residual_threshold
            )

            # ==========================================
            # FINAL CUT DECISION
            # ==========================================

            cut_detected = (

                (
                    not is_monochrome and
                    strong_residual and
                    strong_hist_change
                )

                or

                (
                    is_monochrome and
                    strong_mean_residual and
                    hist_difference > 0.32
                )
            )

            # ==========================================
            # COOLDOWN
            # ==========================================

            if cut_detected:

                if index - last_cut_frame > 20:

                    detected_cuts.append(index)

                    last_cut_frame = index

                    print(
                        f"[CUT DETECTED] "
                        f"Frame {index} | "
                        f"Entropy = {entropy_residual:.2f} | "
                        f"HistDiff = {hist_difference:.2f}"
                    )

        previous_hist_difference = hist_difference

        previous_hist = current_hist.copy()

        # ==============================================
        # RESIDUAL NORMALIZATION
        # ==============================================

        residual_display = cv2.normalize(
            residual_image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        residual_display = residual_display.astype(
            np.uint8
        )

        # ==============================================
        # SAVE REPRESENTATIVE FIGURES
        # ==============================================

        if index == 300:

            current_rgb = cv2.cvtColor(
                frame2,
                cv2.COLOR_BGR2RGB
            )

            predicted_rgb = cv2.cvtColor(
                predicted_frame,
                cv2.COLOR_GRAY2RGB
            )

            residual_rgb = cv2.cvtColor(
                residual_display,
                cv2.COLOR_GRAY2RGB
            )

            fig, axs = plt.subplots(
                1,
                3,
                figsize=(18,5)
            )

            # Current frame

            axs[0].imshow(current_rgb)

            axs[0].set_title(
                f'Current Frame ({index})'
            )

            axs[0].axis('off')

            # Predicted frame

            axs[1].imshow(predicted_rgb)

            axs[1].set_title(
                'Predicted Frame'
            )

            axs[1].axis('off')

            # Residual image

            axs[2].imshow(
                residual_rgb,
                cmap='gray'
            )

            axs[2].set_title(
                'Residual Error Image'
            )

            axs[2].axis('off')

            plt.tight_layout()

            plt.savefig(
                f'../plots/{video_name}_residual_300.png',
                dpi=150
            )

            plt.close()

        # ==============================================
        # DISPLAY INFORMATION
        # ==============================================

        display_frame = frame2.copy()

        cv2.putText(
            display_frame,
            f"Frame : {index}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_frame,
            f"Mean Residual : {mean_residual:.2f}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Residual Entropy : {entropy_residual:.2f}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Hist Difference : {hist_difference:.2f}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # ==============================================
        # LIVE VISUALIZATION
        # ==============================================

        cv2.imshow(
            'Current Video Frame',
            display_frame
        )

        cv2.imshow(
            'Residual Error Image (Rt)',
            residual_display
        )

        # ESC key to stop

        if cv2.waitKey(1) & 0xFF == 27:

            print(
                "  [INFO] Video skipped by user."
            )

            break

        # ==============================================
        # NEXT FRAME
        # ==============================================

        previous_gray = current_gray

        ret, frame2 = cap.read()

        if ret:

            current_gray = cv2.cvtColor(
                frame2,
                cv2.COLOR_BGR2GRAY
            )

    # ==================================================
    # END VIDEO
    # ==================================================

    cap.release()

    cv2.destroyAllWindows()

    print(f"Total frames processed : {index}")

    print(f"Detected cuts : {len(detected_cuts)}")

    print(f"Cut positions : {detected_cuts}")

    # ==================================================
    # STORE RESULTS
    # ==================================================

    all_results.append({

        'video': video_file,
        'frames': frame_indices,
        'mean_residuals': mean_residuals,
        'residual_entropies': residual_entropies,
        'histogram_differences': histogram_differences,
        'detected_cuts': detected_cuts
    })

# ==================================================
# FINAL STATISTICAL PLOTS
# ==================================================

fig, axs = plt.subplots(
    len(all_results),
    3,
    figsize=(18, 4 * len(all_results)),
    sharex=False
)

# Handle single video case

if len(all_results) == 1:
    axs = [axs]

# ==================================================
# PLOT RESULTS
# ==================================================

for i, result in enumerate(all_results):

    frames = result['frames']

    video_name = result[
        'video'
    ].split('/')[-1]

    # ==============================================
    # 1. MEAN RESIDUAL ERROR
    # ==============================================

    axs[i][0].plot(
        frames,
        result['mean_residuals'],
        color='purple',
        linewidth=1.5
    )

    axs[i][0].set_title(
        f'{video_name} : Mean Residual Error'
    )

    axs[i][0].set_xlabel('Frame')
    axs[i][0].set_ylabel('Mean(Rt)')
    axs[i][0].grid(True)

    # ==============================================
    # 2. RESIDUAL ENTROPY
    # ==============================================

    axs[i][1].plot(
        frames,
        result['residual_entropies'],
        color='crimson',
        linewidth=1.5
    )

    axs[i][1].set_title(
        f'{video_name} : Residual Entropy'
    )

    axs[i][1].set_xlabel('Frame')
    axs[i][1].set_ylabel('Entropy(Rt)')
    axs[i][1].grid(True)

    # ==============================================
    # 3. HISTOGRAM DIFFERENCE
    # ==============================================

    axs[i][2].plot(
        frames,
        result['histogram_differences'],
        color='darkblue',
        linewidth=1.5
    )

    axs[i][2].set_title(
        f'{video_name} : Histogram Difference'
    )

    axs[i][2].set_xlabel('Frame')
    axs[i][2].set_ylabel('Bhattacharyya')
    axs[i][2].grid(True)

    # ==============================================
    # DISPLAY DETECTED CUTS
    # ==============================================

    for cut in result['detected_cuts']:

        axs[i][0].axvline(
            cut,
            color='red',
            linestyle='--',
            alpha=0.5
        )

        axs[i][1].axvline(
            cut,
            color='red',
            linestyle='--',
            alpha=0.5
        )

        axs[i][2].axvline(
            cut,
            color='red',
            linestyle='--',
            alpha=0.5
        )

# ==================================================
# FINAL DISPLAY
# ==================================================

plt.tight_layout()

plt.savefig(
    '../plots/q3_statistics.png',
    dpi=150
)

plt.show()