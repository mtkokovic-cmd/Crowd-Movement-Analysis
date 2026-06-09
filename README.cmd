# Crowd Movement Analysis

Computer Vision project for multi-person tracking and crowd behavior analysis using YOLOv11, ByteTrack and Kalman Filtering.

## Preview

### Crowd Movement Heatmap

![Heatmap](crowd_heatmap.png)

### Tracking Example

![Tracking Example](frames/frame_050.jpg)

---

## Overview

This project analyzes pedestrian movement in a crowded urban environment using modern computer vision techniques.

The system detects and tracks multiple people throughout a video sequence, extracts movement trajectories, computes crowd statistics, and generates visualizations that help understand movement patterns.

The original video contains **795 frames**. For computational efficiency, **100 uniformly sampled frames** were processed and analyzed.

---

## Features

- Person detection using YOLOv11
- Multi-object tracking using ByteTrack
- Trajectory extraction and analysis
- Kalman filter trajectory smoothing
- Crowd statistics computation
- People counting over time
- Crowd movement heatmap generation
- CSV export of tracking data
- Annotated video generation

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

## Installation

Clone the repository:

```bash
git clone https://github.com/mtkokovic-cmd/Crowd-Movement-Analysis.git
cd Crowd-Movement-Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

## Processing Pipeline

1. Load the input video.
2. Uniformly sample 100 frames from the original sequence.
3. Detect pedestrians using YOLOv11.
4. Track detected people using ByteTrack.
5. Extract center coordinates of each tracked object.
6. Save trajectory data into CSV files.
7. Apply Kalman filtering for trajectory smoothing.
8. Compute movement statistics.
9. Generate visualizations and reports.

---

## Results

### Crowd Statistics

- Total video frames: 795
- Processed frames: 100
- Maximum detected people: 11
- Average detected people: 7.32
- Valid trajectory segments: 19

### Key Findings

- Longest trajectory segment: 1386.13 px
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

The system automatically generates:

- `output_video.mp4`
- `trajectories.csv`
- `person_summary.csv`
- `people_per_frame.csv`
- `trajectory_plot.png`
- `kalman_trajectory_plot.png`
- `crowd_heatmap.png`
- `people_count_plot.png`
- `frames/` (100 processed frames)

---

## Limitations

Since only 100 uniformly sampled frames were processed, temporary ID switching may occur when individuals are lost and re-detected by the tracker.

For this reason, the reported results refer to trajectory segments rather than guaranteed unique individuals.

---

## Future Improvements

- DeepSORT integration
- Person re-identification (ReID)
- Real-world speed estimation
- Crowd density estimation
- Pedestrian flow analysis

---

## Author

**Matea Koković**

Faculty of Electrical Engineering (ETF)  
University of Belgrade