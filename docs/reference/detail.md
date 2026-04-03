                 ┌─────────────────────────────┐
                 │            INPUT            │
                 │                             │
                 │   Live Webcam Video Stream  │
                 │   (Continuous Video Frames) │
                 └──────────────┬──────────────┘
                                │
                                ▼
              ┌─────────────────────────────────┐
              │        Frame Capture Layer       │
              │   • Webcam captures live frames  │
              │   • Frames passed to processing  │
              └──────────────┬──────────────────┘
                             │
                             ▼
        ┌──────────────────────────────────────────┐
        │     MediaPipe Holistic Landmark Model    │
        │                                          │
        │  Detects keypoints from three regions:   │
        │  • Pose Landmarks (body joints)          │
        │  • Face Landmarks (facial keypoints)     │
        │  • Hand Landmarks (finger joints)        │
        └──────────────┬───────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────┐
        │       Skeletal Landmark Visualization    │
        │                                          │
        │  Landmarks are drawn as a skeletal       │
        │  structure over the video frame to show  │
        │  body pose, hand movement, and face      │
        │  tracking in real time.                  │
        └──────────────┬───────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────┐
        │      Coordinate Normalization Stage      │
        │                                          │
        │  • Extract (x, y, z) coordinates         │
        │  • Normalize landmark values             │
        │  • Prepare consistent feature format     │
        └──────────────┬───────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────┐
        │        Feature Vector Generation         │
        │                                          │
        │  All landmark coordinates are combined   │
        │  into a structured numerical vector      │
        │  representing the spatial configuration  │
        │  of the signer in the frame.             │
        └──────────────┬───────────────────────────┘
                       │
                       ▼
                 ┌─────────────────────────────┐
                 │            OUTPUT            │
                 │                              │
                 │  High-Dimensional Landmark   │
                 │  Feature Vector (≈ 411 dims) │
                 │                              │
                 │  Input for Gesture           │
                 │  Recognition Module (LSTM)   │
                 └─────────────────────────────┘