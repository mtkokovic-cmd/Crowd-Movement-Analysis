# Crowd Movement Analysis

Computer Vision project for multi-person tracking and crowd behavior analysis using YOLOv11, ByteTrack and Kalman Filtering.

## Overview

This project analyzes pedestrian movement in a crowded scene by:

- Detecting people using YOLOv11
- Tracking individuals using ByteTrack
- Extracting movement trajectories
- Applying Kalman filtering for trajectory smoothing
- Computing crowd statistics
- Generating movement visualizations and heatmaps

The original video contains 795 frames. For computational efficiency, 100 uniformly sampled frames were processed and analyzed.

---

## Technologies

- Python
- OpenCV
- YOLOv11 (Ultralytics)
- ByteTrack
- NumPy
- Pandas
- Matplotlib

---

## Processing Pipeline

1. Load video.
2. Uniformly sample 100 frames.
3. Detect people using YOLOv11.
4. Track individuals using ByteTrack.
5. Extract trajectory coordinates.
6. Apply Kalman filtering.
7. Compute movement statistics.
8. Generate visualizations and CSV reports.

---

## Results

### Crowd Statistics

- Total video frames: 795
- Processed frames: 100
- Maximum detected people: 11
- Average detected people: 7.32
- Valid trajectory segments: 19

### Additional Metrics

- Longest trajectory: 1386.13 px
- Fastest trajectory segment: 165.52 px/s

---

## Visualizations

### People Count Over Time

![People Count](people_count_plot.png)

### Raw Trajectories

![Raw Trajectories](trajectory_plot.png)

### Kalman Smoothed Trajectories

![Kalman Trajectories](kalman_trajectory_plot.png)

### Crowd Movement Heatmap

![Heatmap](crowd_heatmap.png)

---

## Generated Outputs

- `output_video.mp4`
- `trajectories.csv`
- `person_summary.csv`
- `people_per_frame.csv`
- `trajectory_plot.png`
- `kalman_trajectory_plot.png`
- `crowd_heatmap.png`
- `people_count_plot.png`

---

## Limitations

Since only 100 sampled frames were processed, ID switching may occur when individuals are temporarily lost and re-detected by the tracker. Therefore, the reported results refer to trajectory segments rather than guaranteed unique individuals.

---

## Author

Matea Koković

Faculty of Electrical Engineering (ETF)
University of Belgrade