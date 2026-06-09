# Angle Measurement Quality

This appendix reports classical measurement quality, which is only a precondition for the frozen-VLM study.

## Aggregate results

- Baseline (`naive_hough`) exact accuracy: 0.00%
- Baseline within-1deg accuracy: 39.58%
- Baseline MAE: 6.332 (95% CI: 2.430, 11.430)
- Scaffold (`vertex_clustered_ray_scaffold`) exact accuracy: 0.00%
- Scaffold within-1deg accuracy: 58.33%
- Scaffold MAE: 2.749 (95% CI: 1.012, 5.832)

## Per-sample results

| Sample | Ground truth | Baseline angle | Scaffold angle | Clutter level | Notes |
|---|---:|---:|---:|---|---|
| angle_001 | 16.2 | 14.7 | 15.9 | low | clutter_segments=1; nuisance_shapes=2; dark_distractors=0; ray_thickness=12 |
| angle_002 | 18.9 | 15.5 | 16.6 | low | clutter_segments=2; nuisance_shapes=3; dark_distractors=0; ray_thickness=12 |
| angle_003 | 21.6 | 21.2 | 21.2 | low | clutter_segments=1; nuisance_shapes=4; dark_distractors=0; ray_thickness=12 |
| angle_004 | 24.3 | 23.7 | 23.4 | low | clutter_segments=2; nuisance_shapes=2; dark_distractors=0; ray_thickness=12 |
| angle_005 | 22.3 | 21.5 | 22.3 | low | clutter_segments=1; nuisance_shapes=3; dark_distractors=0; ray_thickness=12 |
| angle_006 | 25.0 | 22.3 | 24.0 | low | clutter_segments=2; nuisance_shapes=4; dark_distractors=0; ray_thickness=12 |
| angle_007 | 27.7 | 27.0 | 27.7 | low | clutter_segments=1; nuisance_shapes=2; dark_distractors=0; ray_thickness=12 |
| angle_008 | 30.4 | 31.6 | 30.7 | low | clutter_segments=2; nuisance_shapes=3; dark_distractors=0; ray_thickness=12 |
| angle_009 | 28.3 | 26.7 | 26.2 | low | clutter_segments=1; nuisance_shapes=4; dark_distractors=0; ray_thickness=12 |
| angle_010 | 31.0 | 28.1 | 29.1 | low | clutter_segments=2; nuisance_shapes=2; dark_distractors=0; ray_thickness=12 |
| angle_011 | 33.8 | 36.0 | 35.0 | low | clutter_segments=1; nuisance_shapes=3; dark_distractors=0; ray_thickness=12 |
| angle_012 | 36.5 | 36.9 | 35.6 | low | clutter_segments=2; nuisance_shapes=4; dark_distractors=0; ray_thickness=12 |
| angle_013 | 34.4 | 33.5 | 34.1 | low | clutter_segments=1; nuisance_shapes=2; dark_distractors=0; ray_thickness=12 |
| angle_014 | 37.1 | 39.0 | 37.1 | low | clutter_segments=2; nuisance_shapes=3; dark_distractors=0; ray_thickness=12 |
| angle_015 | 39.8 | 141.5 | 37.8 | low | clutter_segments=1; nuisance_shapes=4; dark_distractors=0; ray_thickness=12 |
| angle_016 | 42.5 | 42.7 | 33.6 | low | clutter_segments=2; nuisance_shapes=2; dark_distractors=0; ray_thickness=12 |
| angle_017 | 40.5 | 41.1 | 41.1 | low | clutter_segments=1; nuisance_shapes=3; dark_distractors=0; ray_thickness=12 |
| angle_018 | 43.2 | 42.0 | 40.8 | low | clutter_segments=2; nuisance_shapes=4; dark_distractors=0; ray_thickness=12 |
| angle_019 | 45.9 | 48.3 | 46.9 | low | clutter_segments=1; nuisance_shapes=2; dark_distractors=0; ray_thickness=12 |
| angle_020 | 48.6 | 48.7 | 48.8 | low | clutter_segments=2; nuisance_shapes=3; dark_distractors=0; ray_thickness=12 |
| angle_021 | 46.5 | 45.4 | 48.0 | low | clutter_segments=1; nuisance_shapes=4; dark_distractors=0; ray_thickness=12 |
| angle_022 | 49.2 | 50.4 | 50.8 | low | clutter_segments=2; nuisance_shapes=2; dark_distractors=0; ray_thickness=12 |
| angle_023 | 51.9 | 50.4 | 51.1 | low | clutter_segments=1; nuisance_shapes=3; dark_distractors=0; ray_thickness=12 |
| angle_024 | 54.7 | 55.3 | 55.5 | low | clutter_segments=2; nuisance_shapes=4; dark_distractors=0; ray_thickness=12 |
| angle_025 | 52.6 | 57.9 | 51.7 | medium | clutter_segments=2; nuisance_shapes=3; dark_distractors=0; ray_thickness=12 |
| angle_026 | 55.3 | 50.4 | 54.9 | medium | clutter_segments=3; nuisance_shapes=4; dark_distractors=0; ray_thickness=12 |
| angle_027 | 58.0 | 56.0 | 56.7 | medium | clutter_segments=2; nuisance_shapes=5; dark_distractors=0; ray_thickness=12 |
| angle_028 | 60.7 | 63.3 | 60.9 | medium | clutter_segments=3; nuisance_shapes=3; dark_distractors=0; ray_thickness=12 |
| angle_029 | 58.6 | 60.2 | 59.6 | medium | clutter_segments=2; nuisance_shapes=4; dark_distractors=0; ray_thickness=12 |
| angle_030 | 61.4 | 59.8 | 61.3 | medium | clutter_segments=3; nuisance_shapes=5; dark_distractors=0; ray_thickness=12 |
| angle_031 | 64.1 | 64.6 | 64.2 | medium | clutter_segments=2; nuisance_shapes=3; dark_distractors=0; ray_thickness=12 |
| angle_032 | 66.8 | 65.2 | 65.2 | medium | clutter_segments=3; nuisance_shapes=4; dark_distractors=0; ray_thickness=12 |
| angle_033 | 64.7 | 65.2 | 64.1 | medium | clutter_segments=2; nuisance_shapes=5; dark_distractors=0; ray_thickness=12 |
| angle_034 | 67.4 | 67.8 | 66.8 | medium | clutter_segments=3; nuisance_shapes=3; dark_distractors=0; ray_thickness=12 |
| angle_035 | 70.1 | 73.4 | 71.9 | medium | clutter_segments=2; nuisance_shapes=4; dark_distractors=0; ray_thickness=12 |
| angle_036 | 72.9 | 72.7 | 73.2 | medium | clutter_segments=3; nuisance_shapes=5; dark_distractors=0; ray_thickness=12 |
| angle_037 | 70.8 | 70.2 | 70.2 | medium | clutter_segments=2; nuisance_shapes=3; dark_distractors=0; ray_thickness=12 |
| angle_038 | 73.5 | 73.7 | 74.1 | medium | clutter_segments=3; nuisance_shapes=4; dark_distractors=0; ray_thickness=12 |
| angle_039 | 76.2 | 75.6 | 75.4 | medium | clutter_segments=2; nuisance_shapes=5; dark_distractors=0; ray_thickness=12 |
| angle_040 | 78.9 | 80.0 | 80.2 | medium | clutter_segments=3; nuisance_shapes=3; dark_distractors=0; ray_thickness=12 |
| angle_041 | 76.8 | 76.6 | 76.9 | medium | clutter_segments=2; nuisance_shapes=4; dark_distractors=0; ray_thickness=12 |
| angle_042 | 79.5 | 78.1 | 79.5 | medium | clutter_segments=3; nuisance_shapes=5; dark_distractors=0; ray_thickness=12 |
| angle_043 | 82.3 | 82.9 | 82.5 | medium | clutter_segments=2; nuisance_shapes=3; dark_distractors=0; ray_thickness=12 |
| angle_044 | 85.0 | 83.6 | 86.0 | medium | clutter_segments=3; nuisance_shapes=4; dark_distractors=0; ray_thickness=12 |
| angle_045 | 82.9 | 83.3 | 83.2 | medium | clutter_segments=2; nuisance_shapes=5; dark_distractors=0; ray_thickness=12 |
| angle_046 | 85.6 | 84.3 | 84.8 | medium | clutter_segments=3; nuisance_shapes=3; dark_distractors=0; ray_thickness=12 |
| angle_047 | 88.3 | 88.1 | 88.9 | medium | clutter_segments=2; nuisance_shapes=4; dark_distractors=0; ray_thickness=12 |
| angle_048 | 91.0 | 90.6 | 90.5 | medium | clutter_segments=3; nuisance_shapes=5; dark_distractors=0; ray_thickness=12 |
| angle_049 | 89.0 | 91.0 | 90.5 | high | clutter_segments=3; nuisance_shapes=4; dark_distractors=1; ray_thickness=11 |
| angle_050 | 91.7 | 90.0 | 92.1 | high | clutter_segments=4; nuisance_shapes=5; dark_distractors=0; ray_thickness=11 |
| angle_051 | 94.4 | 93.4 | 93.4 | high | clutter_segments=3; nuisance_shapes=6; dark_distractors=0; ray_thickness=11 |
| angle_052 | 97.1 | 95.3 | 95.3 | high | clutter_segments=4; nuisance_shapes=4; dark_distractors=1; ray_thickness=11 |
| angle_053 | 95.0 | 99.4 | 97.6 | high | clutter_segments=3; nuisance_shapes=5; dark_distractors=0; ray_thickness=11 |
| angle_054 | 97.7 | 98.8 | 99.3 | high | clutter_segments=4; nuisance_shapes=6; dark_distractors=0; ray_thickness=11 |
| angle_055 | 100.5 | 101.4 | 101.3 | high | clutter_segments=3; nuisance_shapes=4; dark_distractors=1; ray_thickness=11 |
| angle_056 | 103.2 | 101.3 | 101.8 | high | clutter_segments=4; nuisance_shapes=5; dark_distractors=0; ray_thickness=11 |
| angle_057 | 101.1 | 102.1 | 100.6 | high | clutter_segments=3; nuisance_shapes=6; dark_distractors=0; ray_thickness=11 |
| angle_058 | 103.8 | 104.4 | 103.9 | high | clutter_segments=4; nuisance_shapes=4; dark_distractors=1; ray_thickness=11 |
| angle_059 | 106.5 | 126.0 | 106.0 | high | clutter_segments=3; nuisance_shapes=5; dark_distractors=0; ray_thickness=11 |
| angle_060 | 109.2 | 109.2 | 109.7 | high | clutter_segments=4; nuisance_shapes=6; dark_distractors=0; ray_thickness=11 |
| angle_061 | 107.1 | 111.1 | 109.6 | high | clutter_segments=3; nuisance_shapes=4; dark_distractors=1; ray_thickness=11 |
| angle_062 | 109.9 | 106.1 | 108.3 | high | clutter_segments=4; nuisance_shapes=5; dark_distractors=0; ray_thickness=11 |
| angle_063 | 112.6 | 112.7 | 112.7 | high | clutter_segments=3; nuisance_shapes=6; dark_distractors=0; ray_thickness=11 |
| angle_064 | 115.3 | 116.2 | 115.8 | high | clutter_segments=4; nuisance_shapes=4; dark_distractors=1; ray_thickness=11 |
| angle_065 | 113.2 | 110.2 | 111.7 | high | clutter_segments=3; nuisance_shapes=5; dark_distractors=0; ray_thickness=11 |
| angle_066 | 115.9 | 114.8 | 116.6 | high | clutter_segments=4; nuisance_shapes=6; dark_distractors=0; ray_thickness=11 |
| angle_067 | 118.6 | 113.2 | 120.4 | high | clutter_segments=3; nuisance_shapes=4; dark_distractors=1; ray_thickness=11 |
| angle_068 | 121.4 | 121.5 | 121.0 | high | clutter_segments=4; nuisance_shapes=5; dark_distractors=0; ray_thickness=11 |
| angle_069 | 119.3 | 120.0 | 119.5 | high | clutter_segments=3; nuisance_shapes=6; dark_distractors=0; ray_thickness=11 |
| angle_070 | 122.0 | 120.6 | 121.1 | high | clutter_segments=4; nuisance_shapes=4; dark_distractors=1; ray_thickness=11 |
| angle_071 | 124.7 | 123.9 | 124.1 | high | clutter_segments=3; nuisance_shapes=5; dark_distractors=0; ray_thickness=11 |
| angle_072 | 127.4 | 126.3 | 126.0 | high | clutter_segments=4; nuisance_shapes=6; dark_distractors=0; ray_thickness=11 |
| angle_073 | 125.3 | 124.8 | 124.8 | extreme | clutter_segments=4; nuisance_shapes=5; dark_distractors=1; ray_thickness=10 |
| angle_074 | 128.1 | 127.1 | 127.0 | extreme | clutter_segments=5; nuisance_shapes=6; dark_distractors=0; ray_thickness=10 |
| angle_075 | 130.8 | 131.9 | 131.0 | extreme | clutter_segments=4; nuisance_shapes=7; dark_distractors=0; ray_thickness=10 |
| angle_076 | 133.5 | 133.1 | 133.5 | extreme | clutter_segments=5; nuisance_shapes=5; dark_distractors=1; ray_thickness=10 |
| angle_077 | 131.4 | 130.5 | 129.9 | extreme | clutter_segments=4; nuisance_shapes=6; dark_distractors=0; ray_thickness=10 |
| angle_078 | 134.1 | 132.8 | 133.0 | extreme | clutter_segments=5; nuisance_shapes=7; dark_distractors=0; ray_thickness=10 |
| angle_079 | 136.8 | 133.9 | 135.0 | extreme | clutter_segments=4; nuisance_shapes=5; dark_distractors=1; ray_thickness=10 |
| angle_080 | 139.5 | 140.1 | 140.7 | extreme | clutter_segments=5; nuisance_shapes=6; dark_distractors=0; ray_thickness=10 |
| angle_081 | 137.5 | 129.7 | 136.7 | extreme | clutter_segments=4; nuisance_shapes=7; dark_distractors=0; ray_thickness=10 |
| angle_082 | 140.2 | 138.2 | 139.2 | extreme | clutter_segments=5; nuisance_shapes=5; dark_distractors=1; ray_thickness=10 |
| angle_083 | 142.9 | 143.9 | 143.1 | extreme | clutter_segments=4; nuisance_shapes=6; dark_distractors=0; ray_thickness=10 |
| angle_084 | 145.6 | 146.0 | 146.9 | extreme | clutter_segments=5; nuisance_shapes=7; dark_distractors=0; ray_thickness=10 |
| angle_085 | 143.5 | 97.9 | 150.2 | extreme | clutter_segments=4; nuisance_shapes=5; dark_distractors=1; ray_thickness=10 |
| angle_086 | 146.2 | 12.8 | 163.8 | extreme | clutter_segments=5; nuisance_shapes=6; dark_distractors=0; ray_thickness=10 |
| angle_087 | 149.0 | 138.3 | 144.9 | extreme | clutter_segments=4; nuisance_shapes=7; dark_distractors=0; ray_thickness=10 |
| angle_088 | 151.7 | 148.6 | 150.3 | extreme | clutter_segments=5; nuisance_shapes=5; dark_distractors=1; ray_thickness=10 |
| angle_089 | 149.6 | 151.6 | 148.5 | extreme | clutter_segments=4; nuisance_shapes=6; dark_distractors=0; ray_thickness=10 |
| angle_090 | 152.3 | 153.8 | 154.2 | extreme | clutter_segments=5; nuisance_shapes=7; dark_distractors=0; ray_thickness=10 |
| angle_091 | 155.0 | 153.4 | 154.6 | extreme | clutter_segments=4; nuisance_shapes=5; dark_distractors=1; ray_thickness=10 |
| angle_092 | 157.7 | 13.2 | 18.8 | extreme | clutter_segments=5; nuisance_shapes=6; dark_distractors=0; ray_thickness=10 |
| angle_093 | 155.7 | 160.9 | 160.1 | extreme | clutter_segments=4; nuisance_shapes=7; dark_distractors=0; ray_thickness=10 |
| angle_094 | 158.4 | 156.4 | 157.3 | extreme | clutter_segments=5; nuisance_shapes=5; dark_distractors=1; ray_thickness=10 |
| angle_095 | 161.1 | 161.9 | 161.6 | extreme | clutter_segments=4; nuisance_shapes=6; dark_distractors=0; ray_thickness=10 |
| angle_096 | 163.8 | 149.7 | 160.5 | extreme | clutter_segments=5; nuisance_shapes=7; dark_distractors=0; ray_thickness=10 |
