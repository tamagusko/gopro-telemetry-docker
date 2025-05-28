import json
import numpy as np
import matplotlib.pyplot as plt

video_name = "GX010018_vid3"

# Load data
with open(f"C:/Users/titou/sdl-colourways-cli/gopro-telemetry-docker/output/{video_name}.json", "r") as f:
    data = json.load(f)

acc_timestamps = np.array(data["ACCL"]["timestamps_s"])
acc_data = np.array(data["ACCL"]["data"])
gyro_timestamps = np.array(data["GYRO"]["timestamps_s"])
gyro_data = np.array(data["GYRO"]["data"])
gps_timestamps = data["GPS9"]["timestamps_s"]
gps_data = data["GPS9"]["data"]

acc_x, acc_y, acc_z = acc_data.T
gyro_x, gyro_y, gyro_z = gyro_data.T

# Avoid altitude bugs
altitudes = [entry[2] if isinstance(entry, list) and len(entry) >= 3 and entry[2] < 8000 else 0 for entry in gps_data]

# Means
mean_acc_x = np.mean(np.abs(acc_x))
mean_acc_y = np.mean(np.abs(acc_y))
mean_acc_z = np.mean(np.abs(acc_z))
mean_gyro_x = np.mean(np.abs(gyro_x))
mean_gyro_y = np.mean(np.abs(gyro_y))
mean_gyro_z = np.mean(np.abs(gyro_z))

# Dynamic thresholds
acc_threshold = 10.0
gyro_threshold = 2.0
window_size = 0.5  # seconds

# Absolute thresholds
abs_min_acc_x = 35
abs_min_acc_z = 35
abs_min_acc_y = 5
abs_min_gyro_x = 0.9
abs_min_gyro_y = 0.9
abs_min_gyro_z = 0.9

def detect_event_windows():
    heavy_braking, bumping, avoidance, stop = [], [], [], []
    n = len(acc_timestamps)

    for i in range(n):
        t = acc_timestamps[i]
        window_indices = [j for j in range(n) if abs(acc_timestamps[j] - t) <= window_size]

        # --- HEAVY BRAKING ---
        if acc_x[i] < -2 * mean_acc_x and acc_x[i] < -abs_min_acc_x:
            az_vals = acc_z[window_indices]
            gz_vals = gyro_z[window_indices]
            if not np.any(az_vals > abs_min_acc_z) and not np.any(gz_vals > abs_min_gyro_z):
                heavy_braking.append((t - window_size, t + window_size))
                continue

        # --- BUMPING ---
        if acc_x[i] < -2 * mean_acc_x and acc_x[i] < -abs_min_acc_x:
            az_vals = acc_z[window_indices]
            gy_vals = gyro_y[window_indices]
            gz_vals = gyro_z[window_indices]
            if (
                np.any(az_vals > 2 * mean_acc_z) and np.max(az_vals) > abs_min_acc_z and
                np.any(gy_vals > 2 * mean_gyro_y) and np.max(gy_vals) > abs_min_gyro_y and np.any(abs(gz_vals) > abs_min_gyro_z)
            ):
                bumping.append((t - window_size, t + window_size))
                continue

        # --- AVOIDANCE ---
        if gyro_x[i] > 2 * mean_gyro_x and gyro_x[i] > abs_min_gyro_x and \
           gyro_z[i] < -2 * mean_gyro_z and gyro_z[i] < -abs_min_gyro_z:
            ay_vals = acc_y[window_indices]
            if np.any(np.abs(ay_vals) > 2 * mean_acc_y) and np.max(np.abs(ay_vals)) > abs_min_acc_y:
                avoidance.append((t - window_size, t + window_size))
        
        # --- STOP ---
        if not gyro_x[i] > 0.5* abs_min_gyro_x and not gyro_z[i] > 0.5* abs_min_gyro_z:
            ay_vals = acc_y[window_indices]
            if not np.max(np.abs(ay_vals)) > 0.5* abs_min_acc_y:
                stop.append((t - window_size, t + window_size))

    return heavy_braking, bumping, avoidance, stop

# Detect events
heavy_braking_events, bumping_events, avoidance_events, stop_event = detect_event_windows()

# Peak detection for the X
acc_x_peaks_idx = np.where(acc_x < -abs_min_acc_x)[0]
acc_z_peaks_idx = np.where(acc_z > abs_min_acc_z)[0]
acc_y_peaks_idx = np.where(np.abs(acc_y) > abs_min_acc_y)[0]
gyro_y_peaks_idx = np.where(gyro_y > abs_min_gyro_y)[0]
gyro_x_peaks_idx = np.where(gyro_x > abs_min_gyro_x)[0]
gyro_z_peaks_idx = np.where(gyro_z < -abs_min_gyro_z)[0]

# ------------------- PLOTS -------------------
plt.figure(figsize=(14, 10))

# ACCELERATION
plt.subplot(3, 1, 1)
plt.plot(acc_timestamps, acc_x, label="Acc X", color='r')
plt.plot(acc_timestamps, acc_y, label="Acc Y", color='g')
plt.plot(acc_timestamps, acc_z, label="Acc Z", color='b')

# Rectangles
for start, end in heavy_braking_events:
    plt.axvspan(start, end, color='red', alpha=0.3, label="Heavy Braking")
for start, end in bumping_events:
    plt.axvspan(start, end, color='blue', alpha=0.3, label="Bumping")
for start, end in avoidance_events:
    plt.axvspan(start, end, color='green', alpha=0.3, label="Avoidance")
for start, end in stop_event:
    plt.axvspan(start, end, color='orange', alpha=0.3, label="Stop")

plt.title("Acceleration Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.grid(True)

# GYROSCOPE
plt.subplot(3, 1, 2)
plt.plot(gyro_timestamps, gyro_x, label="Gyro X", color='orange')
plt.plot(gyro_timestamps, gyro_y, label="Gyro Y", color='purple')
plt.plot(gyro_timestamps, gyro_z, label="Gyro Z", color='cyan')

# Rectangles
for start, end in heavy_braking_events:
    plt.axvspan(start, end, color='red', alpha=0.3)
for start, end in bumping_events:
    plt.axvspan(start, end, color='blue', alpha=0.3)
for start, end in avoidance_events:
    plt.axvspan(start, end, color='green', alpha=0.3)
for start, end in stop_event:
    plt.axvspan(start, end, color='orange', alpha=0.3, label="Stop")    

plt.title("Gyroscope Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (°/s)")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.grid(True)

# ALTITUDE
plt.subplot(3, 1, 3)
plt.plot(gps_timestamps[:len(altitudes)], altitudes, label="Altitude", color='brown')
plt.title("Altitude Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
