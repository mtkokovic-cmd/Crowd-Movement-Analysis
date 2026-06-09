import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO

# -----------------------------
# 1. Settings
# -----------------------------

VIDEO_PATH = "videos/crowd.mp4"

OUTPUT_VIDEO_PATH = "output_video.mp4"
TRAJECTORIES_CSV = "trajectories.csv"
PERSON_SUMMARY_CSV = "person_summary.csv"
PEOPLE_COUNT_CSV = "people_per_frame.csv"

RAW_PLOT_PATH = "trajectory_plot.png"
KALMAN_PLOT_PATH = "kalman_trajectory_plot.png"
HEATMAP_PATH = "crowd_heatmap.png"
COUNT_PLOT_PATH = "people_count_plot.png"

FRAMES_DIR = "frames"

NUM_FRAMES_TO_PROCESS = 100
MIN_TRAJECTORY_LENGTH = 15

os.makedirs(FRAMES_DIR, exist_ok=True)

# -----------------------------
# 2. Load YOLO
# -----------------------------

model = YOLO("yolo11n.pt")

# -----------------------------
# 3. Kalman filter
# -----------------------------

def apply_kalman_filter(points):
    if len(points) < 2:
        return points

    kalman = cv2.KalmanFilter(4, 2)

    kalman.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], np.float32)

    kalman.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], np.float32)

    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
    kalman.errorCovPost = np.eye(4, dtype=np.float32)

    first_x, first_y = points[0]

    kalman.statePost = np.array([
        [np.float32(first_x)],
        [np.float32(first_y)],
        [0],
        [0]
    ], np.float32)

    smoothed_points = []

    for x, y in points:
        measurement = np.array([
            [np.float32(x)],
            [np.float32(y)]
        ])

        kalman.predict()
        corrected = kalman.correct(measurement)

        smoothed_x = int(corrected[0, 0])
        smoothed_y = int(corrected[1, 0])

        smoothed_points.append((smoothed_x, smoothed_y))

    return smoothed_points

# -----------------------------
# 4. Open video
# -----------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

selected_frames = np.linspace(
    0,
    total_frames - 1,
    NUM_FRAMES_TO_PROCESS
).astype(int)

selected_frames_set = set(selected_frames)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

print(f"Total frames in video: {total_frames}")
print(f"Processing {NUM_FRAMES_TO_PROCESS} sampled frames...")

# -----------------------------
# 5. Tracking storage
# -----------------------------

track_history = defaultdict(list)
trajectory_rows = []
people_count_rows = []

frame_number = 0
processed_count = 0

# -----------------------------
# 6. Process only sampled frames
# -----------------------------

while True:
    success, frame = cap.read()

    if not success:
        break

    if frame_number not in selected_frames_set:
        frame_number += 1
        continue

    processed_count += 1

    results = model.track(
        frame,
        persist=True,
        classes=[0],          # person class
        tracker="bytetrack.yaml",
        conf=0.45,
        verbose=False
    )

    annotated_frame = frame.copy()
    people_in_frame = 0

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        people_in_frame = len(track_ids)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            track_history[track_id].append((center_x, center_y))

            time_s = frame_number / fps

            trajectory_rows.append({
                "sample_index": processed_count,
                "frame": frame_number,
                "time_s": round(time_s, 2),
                "track_id": track_id,
                "x_center": center_x,
                "y_center": center_y
            })

            cv2.rectangle(
                annotated_frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                f"ID {track_id}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            points = track_history[track_id]

            for i in range(1, len(points)):
                cv2.line(
                    annotated_frame,
                    points[i - 1],
                    points[i],
                    (0, 0, 255),
                    2
                )

    people_count_rows.append({
        "sample_index": processed_count,
        "frame": frame_number,
        "time_s": round(frame_number / fps, 2),
        "people_count": people_in_frame
    })

    frame_path = os.path.join(FRAMES_DIR, f"frame_{processed_count:03d}.jpg")
    cv2.imwrite(frame_path, annotated_frame)

    out.write(annotated_frame)

    frame_number += 1

cap.release()
out.release()

# -----------------------------
# 7. Save trajectories
# -----------------------------

df = pd.DataFrame(trajectory_rows)

if df.empty:
    print("No people detected.")
    exit()

valid_ids = []

for track_id in sorted(df["track_id"].unique()):
    person_data = df[df["track_id"] == track_id]

    if len(person_data) >= MIN_TRAJECTORY_LENGTH:
        valid_ids.append(track_id)

id_mapping = {
    old_id: new_id + 1
    for new_id, old_id in enumerate(valid_ids)
}

df["person_id"] = df["track_id"].map(id_mapping)
df = df.dropna(subset=["person_id"])
df["person_id"] = df["person_id"].astype(int)

df.to_csv(TRAJECTORIES_CSV, index=False)

count_df = pd.DataFrame(people_count_rows)
count_df.to_csv(PEOPLE_COUNT_CSV, index=False)

print("\nOriginal tracker IDs:")
print(valid_ids)

print("\nPerson mapping:")
print(id_mapping)

print(f"\nTotal valid trajectory segments: {len(valid_ids)}")

# -----------------------------
# 8. Person summary
# -----------------------------

summary_rows = []

print("\nPerson movement summary:")

for person_id in sorted(df["person_id"].unique()):
    person_data = df[df["person_id"] == person_id].sort_values("time_s")

    start_time = person_data["time_s"].min()
    end_time = person_data["time_s"].max()
    duration = end_time - start_time

    points = list(zip(
        person_data["x_center"].values,
        person_data["y_center"].values
    ))

    total_distance = 0

    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]

        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_distance += distance

    average_speed = total_distance / duration if duration > 0 else 0

    summary_rows.append({
        "person_id": person_id,
        "start_time_s": round(start_time, 2),
        "end_time_s": round(end_time, 2),
        "duration_s": round(duration, 2),
        "total_distance_px": round(total_distance, 2),
        "average_speed_px_per_s": round(average_speed, 2)
    })

    print(
        f"Person {person_id}: "
        f"{start_time:.2f}s - {end_time:.2f}s | "
        f"duration: {duration:.2f}s | "
        f"distance: {total_distance:.2f}px | "
        f"avg speed: {average_speed:.2f}px/s"
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(PERSON_SUMMARY_CSV, index=False)

# -----------------------------
# Crowd statistics
# -----------------------------

max_people = count_df["people_count"].max()
avg_people = count_df["people_count"].mean()

fastest_person = summary_df.loc[
    summary_df["average_speed_px_per_s"].idxmax()
]

longest_person = summary_df.loc[
    summary_df["total_distance_px"].idxmax()
]

print("\nCrowd statistics")
print("-----------------------------")

print(f"Maximum people detected: {max_people}")
print(f"Average people detected: {avg_people:.2f}")

print(
    f"Fastest person: Person {int(fastest_person['person_id'])} "
    f"({fastest_person['average_speed_px_per_s']:.2f} px/s)"
)

print(
    f"Longest trajectory: Person {int(longest_person['person_id'])} "
    f"({longest_person['total_distance_px']:.2f} px)"
)
# -----------------------------
# 9. Raw trajectory plot
# -----------------------------

plt.figure(figsize=(10, 6))

for person_id in sorted(df["person_id"].unique()):
    person_data = df[df["person_id"] == person_id]

    plt.plot(
        person_data["x_center"],
        person_data["y_center"],
        marker="o",
        markersize=2,
        linewidth=2,
        label=f"Person {person_id}"
    )

plt.gca().invert_yaxis()
plt.title("Raw People Trajectories")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.legend()
plt.grid(True)
plt.savefig(RAW_PLOT_PATH)
plt.close()

# -----------------------------
# 10. Kalman trajectory plot
# -----------------------------

plt.figure(figsize=(10, 6))

for person_id in sorted(df["person_id"].unique()):
    person_data = df[df["person_id"] == person_id]

    raw_points = list(zip(
        person_data["x_center"].values,
        person_data["y_center"].values
    ))

    smoothed_points = apply_kalman_filter(raw_points)

    xs = [p[0] for p in smoothed_points]
    ys = [p[1] for p in smoothed_points]

    plt.plot(
        xs,
        ys,
        marker="o",
        markersize=2,
        linewidth=2,
        label=f"Person {person_id}"
    )

plt.gca().invert_yaxis()
plt.title("Kalman Smoothed People Trajectories")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.legend()
plt.grid(True)
plt.savefig(KALMAN_PLOT_PATH)
plt.close()

# -----------------------------
# 11. People count plot
# -----------------------------

plt.figure(figsize=(10, 5))
plt.plot(
    count_df["time_s"],
    count_df["people_count"],
    marker="o",
    linewidth=2
)
plt.title("People Count Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Number of detected people")
plt.grid(True)
plt.savefig(COUNT_PLOT_PATH)
plt.close()

# -----------------------------
# 12. Crowd heatmap
# -----------------------------

heatmap = np.zeros((height, width), dtype=np.float32)

for _, row in df.iterrows():
    x = int(row["x_center"])
    y = int(row["y_center"])

    if 0 <= x < width and 0 <= y < height:
        heatmap[y, x] += 1

heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=40, sigmaY=40)
heatmap = heatmap / heatmap.max()

plt.figure(figsize=(10, 6))
plt.imshow(heatmap, cmap="hot")
plt.title("Crowd Movement Heatmap")
plt.axis("off")
plt.savefig(HEATMAP_PATH)
plt.close()

print("\nFiles created:")
print(OUTPUT_VIDEO_PATH)
print(TRAJECTORIES_CSV)
print(PERSON_SUMMARY_CSV)
print(PEOPLE_COUNT_CSV)
print(RAW_PLOT_PATH)
print(KALMAN_PLOT_PATH)
print(COUNT_PLOT_PATH)
print(HEATMAP_PATH)
print(FRAMES_DIR)