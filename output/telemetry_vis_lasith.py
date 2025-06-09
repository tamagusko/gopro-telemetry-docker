import json
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from numba import jit
from scipy.ndimage import uniform_filter1d

@jit(nopython=True)
def detect_event_windows_optimized(acc_timestamps, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, 
                                 window_size, mean_acc_x, mean_acc_y, mean_acc_z, 
                                 mean_gyro_x, mean_gyro_y, mean_gyro_z,
                                 abs_min_acc_x, abs_min_acc_z, abs_min_acc_y, 
                                 abs_min_gyro_x, abs_min_gyro_y, abs_min_gyro_z):
    """Optimized event detection using sliding window approach with numba JIT compilation"""
    n = len(acc_timestamps)
    heavy_braking = []
    bumping = []
    avoidance = []
    stop = []
    
    # Pre-compute window size in terms of array indices (assuming roughly uniform sampling)
    if n > 1:
        avg_dt = (acc_timestamps[-1] - acc_timestamps[0]) / (n - 1)
        window_idx_size = int(window_size / avg_dt)
    else:
        window_idx_size = 1
    
    for i in range(n):
        t = acc_timestamps[i]
        
        # Define window bounds more efficiently
        start_idx = max(0, i - window_idx_size)
        end_idx = min(n, i + window_idx_size + 1)
        
        # --- HEAVY BRAKING ---
        if acc_x[i] < -2 * mean_acc_x and acc_x[i] < -abs_min_acc_x:
            az_max = np.max(acc_z[start_idx:end_idx])
            gz_max = np.max(gyro_z[start_idx:end_idx])
            if az_max <= abs_min_acc_z and gz_max <= abs_min_gyro_z:
                heavy_braking.append((t - window_size, t + window_size))
                continue

        # --- BUMPING ---
        if acc_x[i] < -2 * mean_acc_x and acc_x[i] < -abs_min_acc_x:
            az_max = np.max(acc_z[start_idx:end_idx])
            gy_max = np.max(gyro_y[start_idx:end_idx])
            gz_abs_max = np.max(np.abs(gyro_z[start_idx:end_idx]))
            
            if (az_max > 2 * mean_acc_z and az_max > abs_min_acc_z and
                gy_max > 2 * mean_gyro_y and gy_max > abs_min_gyro_y and 
                gz_abs_max > abs_min_gyro_z):
                bumping.append((t - window_size, t + window_size))
                continue

        # --- AVOIDANCE ---
        if (gyro_x[i] > 2 * mean_gyro_x and gyro_x[i] > abs_min_gyro_x and
            gyro_z[i] < -2 * mean_gyro_z and gyro_z[i] < -abs_min_gyro_z):
            ay_abs_max = np.max(np.abs(acc_y[start_idx:end_idx]))
            if ay_abs_max > 2 * mean_acc_y and ay_abs_max > abs_min_acc_y:
                avoidance.append((t - window_size, t + window_size))
        
        # --- STOP ---
        if (gyro_x[i] <= 0.5 * abs_min_gyro_x and gyro_z[i] <= 0.5 * abs_min_gyro_z):
            ay_abs_max = np.max(np.abs(acc_y[start_idx:end_idx]))
            if ay_abs_max <= 0.5 * abs_min_acc_y:
                stop.append((t - window_size, t + window_size))

    return heavy_braking, bumping, avoidance, stop

@jit(nopython=True)
def detect_peaks_optimized(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, 
                          abs_min_acc_x, abs_min_acc_y, abs_min_acc_z, 
                          abs_min_gyro_x, abs_min_gyro_y, abs_min_gyro_z):
    """Optimized peak detection using vectorized operations with numba"""
    acc_x_peaks_idx = np.where(acc_x < -abs_min_acc_x)[0]
    acc_z_peaks_idx = np.where(acc_z > abs_min_acc_z)[0]
    acc_y_peaks_idx = np.where(np.abs(acc_y) > abs_min_acc_y)[0]
    gyro_y_peaks_idx = np.where(gyro_y > abs_min_gyro_y)[0]
    gyro_x_peaks_idx = np.where(gyro_x > abs_min_gyro_x)[0]
    gyro_z_peaks_idx = np.where(gyro_z < -abs_min_gyro_z)[0]
    return acc_x_peaks_idx, acc_z_peaks_idx, acc_y_peaks_idx, gyro_y_peaks_idx, gyro_x_peaks_idx, gyro_z_peaks_idx

def preprocess_gps_data(gps_data):
    """Optimized GPS data preprocessing using vectorized operations"""
    gps_array = np.array(gps_data)
    if gps_array.ndim == 2 and gps_array.shape[1] >= 3:
        altitudes = gps_array[:, 2]
        altitudes = np.where(altitudes < 8000, altitudes, 0)
    else:
        # Fallback for irregular data
        altitudes = []
        for entry in gps_data:
            if isinstance(entry, list) and len(entry) >= 3 and entry[2] < 8000:
                altitudes.append(entry[2])
            else:
                altitudes.append(0)
        altitudes = np.array(altitudes)
    return altitudes

def merge_overlapping_events(events, merge_threshold=2.0):
    """Merge overlapping or closely spaced events to reduce noise"""
    if not events:
        return events
    
    events = sorted(events)
    merged = [events[0]]
    
    for current in events[1:]:
        last = merged[-1]
        # If current event starts within merge_threshold of last event's end
        if current[0] <= last[1] + merge_threshold:
            # Merge by extending the end time
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    
    return merged

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process telemetry data for event detection.")
    parser.add_argument("--file", type=str, default="GX010001_PhoenixPark.json", 
                       help="Name of the video file without extension")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting for faster execution")
    args = parser.parse_args()
    
    video_name = args.file
    print(f"[INFO] Processing video: {video_name}")
    
    stime = time.time()
    
    # Load and preprocess data
    print("[INFO] Loading data...")
    with open(f"./{video_name}", "r") as f:
        data = json.load(f)

    # Convert to numpy arrays immediately for better performance
    acc_timestamps = np.array(data["ACCL"]["timestamps_s"], dtype=np.float64)
    acc_data = np.array(data["ACCL"]["data"], dtype=np.float64)
    gyro_timestamps = np.array(data["GYRO"]["timestamps_s"], dtype=np.float64)
    gyro_data = np.array(data["GYRO"]["data"], dtype=np.float64)
    gps_timestamps = np.array(data["GPS9"]["timestamps_s"], dtype=np.float64)
    
    acc_x, acc_y, acc_z = acc_data.T
    gyro_x, gyro_y, gyro_z = gyro_data.T

    print("[INFO] Preprocessing GPS data...")
    altitudes = preprocess_gps_data(data["GPS9"]["data"])

    print("[INFO] Computing statistics...")
    # Vectorized mean calculations
    mean_acc_x = np.mean(np.abs(acc_x))
    mean_acc_y = np.mean(np.abs(acc_y))
    mean_acc_z = np.mean(np.abs(acc_z))
    mean_gyro_x = np.mean(np.abs(gyro_x))
    mean_gyro_y = np.mean(np.abs(gyro_y))
    mean_gyro_z = np.mean(np.abs(gyro_z))

    # Parameters
    window_size = 1.0  # seconds
    abs_min_acc_x = 35
    abs_min_acc_z = 35
    abs_min_acc_y = 5
    abs_min_gyro_x = 0.9
    abs_min_gyro_y = 0.9
    abs_min_gyro_z = 0.9

    print("[INFO] Detecting events and peaks...")
    # Use optimized functions with multithreading
    with ThreadPoolExecutor(max_workers=2) as executor:
        event_future = executor.submit(
            detect_event_windows_optimized,
            acc_timestamps, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, window_size,
            mean_acc_x, mean_acc_y, mean_acc_z, mean_gyro_x, mean_gyro_y, mean_gyro_z,
            abs_min_acc_x, abs_min_acc_z, abs_min_acc_y, abs_min_gyro_x, abs_min_gyro_y, abs_min_gyro_z
        )
        peaks_future = executor.submit(
            detect_peaks_optimized,
            acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z,
            abs_min_acc_x, abs_min_acc_y, abs_min_acc_z, abs_min_gyro_x, abs_min_gyro_y, abs_min_gyro_z
        )
        
        heavy_braking_events, bumping_events, avoidance_events, stop_events = event_future.result()
        acc_x_peaks_idx, acc_z_peaks_idx, acc_y_peaks_idx, gyro_y_peaks_idx, gyro_x_peaks_idx, gyro_z_peaks_idx = peaks_future.result()

    # Merge overlapping events to reduce noise
    print("[INFO] Post-processing events...")
    heavy_braking_events = merge_overlapping_events(heavy_braking_events)
    bumping_events = merge_overlapping_events(bumping_events)
    avoidance_events = merge_overlapping_events(avoidance_events)
    stop_events = merge_overlapping_events(stop_events)

    # Print summary
    print(f"[RESULTS] Heavy braking events: {len(heavy_braking_events)}")
    print(f"[RESULTS] Bumping events: {len(bumping_events)}")
    print(f"[RESULTS] Avoidance events: {len(avoidance_events)}")
    print(f"[RESULTS] Stop events: {len(stop_events)}")

    if not args.no_plot:
        print("[INFO] Generating plots...")
        # ------------------- PLOTS -------------------
        plt.figure(figsize=(14, 10))

        # ACCELERATION
        plt.subplot(3, 1, 1)
        plt.plot(acc_timestamps, acc_x, label="Acc X", color='r', linewidth=0.8)
        plt.plot(acc_timestamps, acc_y, label="Acc Y", color='g', linewidth=0.8)
        plt.plot(acc_timestamps, acc_z, label="Acc Z", color='b', linewidth=0.8)

        # Plot events with unique labels
        event_colors = {'Heavy Braking': 'red', 'Bumping': 'blue', 'Avoidance': 'green', 'Stop': 'orange'}
        plotted_labels = set()
        
        for events, label, color in [(heavy_braking_events, 'Heavy Braking', 'red'),
                                   (bumping_events, 'Bumping', 'blue'),
                                   (avoidance_events, 'Avoidance', 'green'),
                                   (stop_events, 'Stop', 'orange')]:
            for start, end in events:
                plt.axvspan(start, end, color=color, alpha=0.3, 
                           label=label if label not in plotted_labels else "")
                plotted_labels.add(label)

        plt.title("Acceleration Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (m/s²)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # GYROSCOPE
        plt.subplot(3, 1, 2)
        plt.plot(gyro_timestamps, gyro_x, label="Gyro X", color='orange', linewidth=0.8)
        plt.plot(gyro_timestamps, gyro_y, label="Gyro Y", color='purple', linewidth=0.8)
        plt.plot(gyro_timestamps, gyro_z, label="Gyro Z", color='cyan', linewidth=0.8)

        # Plot events (no labels needed here)
        for events, color in [(heavy_braking_events, 'red'), (bumping_events, 'blue'),
                             (avoidance_events, 'green'), (stop_events, 'orange')]:
            for start, end in events:
                plt.axvspan(start, end, color=color, alpha=0.3)

        plt.title("Gyroscope Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Velocity (°/s)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # ALTITUDE
        plt.subplot(3, 1, 3)
        plt.plot(gps_timestamps[:len(altitudes)], altitudes, label="Altitude", color='brown', linewidth=0.8)
        plt.title("Altitude Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Altitude (m)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    etime = time.time()
    print(f"[INFO] Total execution time: {etime - stime:.2f} seconds")