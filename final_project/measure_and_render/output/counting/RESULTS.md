# Counting Measurement Quality

This appendix reports classical measurement quality, which is only a precondition for the frozen-VLM study.

## Aggregate results

- Baseline (`connected_components`) exact accuracy: 1.04%
- Baseline MAE: 3.354 (95% CI: 3.062, 3.646)
- Scaffold (`watershed`) exact accuracy: 93.75%
- Scaffold MAE: 0.062 (95% CI: 0.021, 0.115)

## Per-sample results

| Sample | Ground truth | CC count | Watershed count | Overlap level | Notes |
|---|---:|---:|---:|---|---|
| count_001 | 6 | 5 | 6 | low | touching_pairs=1; distractors=10; occluders=1; texture_strength=6 |
| count_002 | 7 | 5 | 7 | low | touching_pairs=2; distractors=11; occluders=0; texture_strength=6 |
| count_003 | 8 | 6 | 8 | low | touching_pairs=2; distractors=12; occluders=0; texture_strength=6 |
| count_004 | 9 | 6 | 9 | low | touching_pairs=3; distractors=13; occluders=0; texture_strength=6 |
| count_005 | 10 | 9 | 10 | low | touching_pairs=1; distractors=14; occluders=0; texture_strength=6 |
| count_006 | 11 | 9 | 11 | low | touching_pairs=2; distractors=10; occluders=0; texture_strength=6 |
| count_007 | 12 | 9 | 12 | low | touching_pairs=3; distractors=11; occluders=1; texture_strength=6 |
| count_008 | 13 | 9 | 13 | low | touching_pairs=4; distractors=12; occluders=0; texture_strength=6 |
| count_009 | 14 | 13 | 14 | low | touching_pairs=1; distractors=13; occluders=0; texture_strength=6 |
| count_010 | 15 | 13 | 15 | low | touching_pairs=2; distractors=14; occluders=0; texture_strength=6 |
| count_011 | 16 | 13 | 16 | low | touching_pairs=3; distractors=10; occluders=0; texture_strength=6 |
| count_012 | 17 | 13 | 17 | low | touching_pairs=4; distractors=11; occluders=0; texture_strength=6 |
| count_013 | 18 | 17 | 18 | low | touching_pairs=1; distractors=12; occluders=1; texture_strength=6 |
| count_014 | 19 | 17 | 19 | low | touching_pairs=2; distractors=13; occluders=0; texture_strength=6 |
| count_015 | 20 | 17 | 20 | low | touching_pairs=3; distractors=14; occluders=0; texture_strength=6 |
| count_016 | 21 | 17 | 21 | low | touching_pairs=4; distractors=10; occluders=0; texture_strength=6 |
| count_017 | 22 | 21 | 22 | low | touching_pairs=1; distractors=11; occluders=0; texture_strength=6 |
| count_018 | 23 | 21 | 23 | low | touching_pairs=2; distractors=12; occluders=0; texture_strength=6 |
| count_019 | 24 | 21 | 24 | low | touching_pairs=3; distractors=13; occluders=1; texture_strength=6 |
| count_020 | 25 | 22 | 25 | low | touching_pairs=4; distractors=14; occluders=0; texture_strength=6 |
| count_021 | 25 | 24 | 25 | low | touching_pairs=1; distractors=10; occluders=0; texture_strength=6 |
| count_022 | 25 | 23 | 25 | low | touching_pairs=2; distractors=11; occluders=0; texture_strength=6 |
| count_023 | 25 | 21 | 24 | low | touching_pairs=3; distractors=12; occluders=0; texture_strength=6 |
| count_024 | 25 | 21 | 25 | low | touching_pairs=4; distractors=13; occluders=0; texture_strength=6 |
| count_025 | 6 | 4 | 6 | medium | touching_pairs=2; distractors=20; occluders=2; texture_strength=9 |
| count_026 | 7 | 6 | 7 | medium | touching_pairs=2; distractors=16; occluders=1; texture_strength=9 |
| count_027 | 8 | 6 | 8 | medium | touching_pairs=2; distractors=17; occluders=1; texture_strength=9 |
| count_028 | 9 | 6 | 9 | medium | touching_pairs=3; distractors=18; occluders=1; texture_strength=9 |
| count_029 | 10 | 8 | 10 | medium | touching_pairs=2; distractors=19; occluders=1; texture_strength=9 |
| count_030 | 11 | 8 | 11 | medium | touching_pairs=3; distractors=20; occluders=1; texture_strength=9 |
| count_031 | 12 | 9 | 12 | medium | touching_pairs=4; distractors=16; occluders=2; texture_strength=9 |
| count_032 | 13 | 9 | 13 | medium | touching_pairs=4; distractors=17; occluders=1; texture_strength=9 |
| count_033 | 14 | 12 | 14 | medium | touching_pairs=2; distractors=18; occluders=1; texture_strength=9 |
| count_034 | 15 | 12 | 15 | medium | touching_pairs=3; distractors=19; occluders=1; texture_strength=9 |
| count_035 | 16 | 13 | 16 | medium | touching_pairs=4; distractors=20; occluders=1; texture_strength=9 |
| count_036 | 17 | 12 | 17 | medium | touching_pairs=5; distractors=16; occluders=1; texture_strength=9 |
| count_037 | 18 | 16 | 18 | medium | touching_pairs=2; distractors=17; occluders=2; texture_strength=9 |
| count_038 | 19 | 16 | 19 | medium | touching_pairs=3; distractors=18; occluders=1; texture_strength=9 |
| count_039 | 20 | 16 | 20 | medium | touching_pairs=4; distractors=19; occluders=1; texture_strength=9 |
| count_040 | 21 | 16 | 21 | medium | touching_pairs=5; distractors=20; occluders=1; texture_strength=9 |
| count_041 | 22 | 19 | 22 | medium | touching_pairs=2; distractors=16; occluders=1; texture_strength=9 |
| count_042 | 23 | 20 | 23 | medium | touching_pairs=3; distractors=17; occluders=1; texture_strength=9 |
| count_043 | 24 | 20 | 24 | medium | touching_pairs=4; distractors=18; occluders=2; texture_strength=9 |
| count_044 | 25 | 20 | 25 | medium | touching_pairs=5; distractors=19; occluders=1; texture_strength=9 |
| count_045 | 25 | 22 | 25 | medium | touching_pairs=2; distractors=20; occluders=1; texture_strength=9 |
| count_046 | 25 | 22 | 25 | medium | touching_pairs=3; distractors=16; occluders=1; texture_strength=9 |
| count_047 | 25 | 21 | 24 | medium | touching_pairs=4; distractors=17; occluders=1; texture_strength=9 |
| count_048 | 25 | 20 | 25 | medium | touching_pairs=5; distractors=18; occluders=1; texture_strength=9 |
| count_049 | 6 | 6 | 6 | high | touching_pairs=2; distractors=25; occluders=3; texture_strength=12 |
| count_050 | 7 | 6 | 7 | high | touching_pairs=2; distractors=26; occluders=2; texture_strength=12 |
| count_051 | 8 | 7 | 8 | high | touching_pairs=2; distractors=22; occluders=2; texture_strength=12 |
| count_052 | 9 | 6 | 9 | high | touching_pairs=3; distractors=23; occluders=2; texture_strength=12 |
| count_053 | 10 | 7 | 10 | high | touching_pairs=3; distractors=24; occluders=2; texture_strength=12 |
| count_054 | 11 | 8 | 11 | high | touching_pairs=3; distractors=25; occluders=2; texture_strength=12 |
| count_055 | 12 | 8 | 12 | high | touching_pairs=4; distractors=26; occluders=3; texture_strength=12 |
| count_056 | 13 | 10 | 13 | high | touching_pairs=4; distractors=22; occluders=2; texture_strength=12 |
| count_057 | 14 | 12 | 14 | high | touching_pairs=3; distractors=23; occluders=2; texture_strength=12 |
| count_058 | 15 | 11 | 15 | high | touching_pairs=4; distractors=24; occluders=2; texture_strength=12 |
| count_059 | 16 | 11 | 16 | high | touching_pairs=5; distractors=25; occluders=2; texture_strength=12 |
| count_060 | 17 | 13 | 17 | high | touching_pairs=5; distractors=26; occluders=2; texture_strength=12 |
| count_061 | 18 | 15 | 18 | high | touching_pairs=3; distractors=22; occluders=3; texture_strength=12 |
| count_062 | 19 | 15 | 19 | high | touching_pairs=4; distractors=23; occluders=2; texture_strength=12 |
| count_063 | 20 | 15 | 20 | high | touching_pairs=5; distractors=24; occluders=2; texture_strength=12 |
| count_064 | 21 | 15 | 21 | high | touching_pairs=6; distractors=25; occluders=2; texture_strength=12 |
| count_065 | 22 | 19 | 22 | high | touching_pairs=3; distractors=26; occluders=2; texture_strength=12 |
| count_066 | 23 | 19 | 23 | high | touching_pairs=4; distractors=22; occluders=2; texture_strength=12 |
| count_067 | 24 | 19 | 24 | high | touching_pairs=5; distractors=23; occluders=3; texture_strength=12 |
| count_068 | 25 | 19 | 25 | high | touching_pairs=6; distractors=24; occluders=2; texture_strength=12 |
| count_069 | 25 | 23 | 25 | high | touching_pairs=3; distractors=25; occluders=2; texture_strength=12 |
| count_070 | 25 | 21 | 25 | high | touching_pairs=4; distractors=26; occluders=2; texture_strength=12 |
| count_071 | 25 | 20 | 25 | high | touching_pairs=5; distractors=22; occluders=2; texture_strength=12 |
| count_072 | 25 | 20 | 25 | high | touching_pairs=6; distractors=23; occluders=2; texture_strength=12 |
| count_073 | 6 | 4 | 6 | extreme | touching_pairs=2; distractors=30; occluders=4; texture_strength=15 |
| count_074 | 7 | 5 | 7 | extreme | touching_pairs=2; distractors=31; occluders=3; texture_strength=15 |
| count_075 | 8 | 7 | 9 | extreme | touching_pairs=2; distractors=32; occluders=3; texture_strength=15 |
| count_076 | 9 | 6 | 9 | extreme | touching_pairs=3; distractors=28; occluders=3; texture_strength=15 |
| count_077 | 10 | 7 | 10 | extreme | touching_pairs=3; distractors=29; occluders=3; texture_strength=15 |
| count_078 | 11 | 8 | 11 | extreme | touching_pairs=3; distractors=30; occluders=3; texture_strength=15 |
| count_079 | 12 | 8 | 13 | extreme | touching_pairs=4; distractors=31; occluders=4; texture_strength=15 |
| count_080 | 13 | 9 | 13 | extreme | touching_pairs=4; distractors=32; occluders=3; texture_strength=15 |
| count_081 | 14 | 11 | 14 | extreme | touching_pairs=4; distractors=28; occluders=3; texture_strength=15 |
| count_082 | 15 | 11 | 15 | extreme | touching_pairs=5; distractors=29; occluders=3; texture_strength=15 |
| count_083 | 16 | 11 | 15 | extreme | touching_pairs=5; distractors=30; occluders=3; texture_strength=15 |
| count_084 | 17 | 11 | 17 | extreme | touching_pairs=5; distractors=31; occluders=3; texture_strength=15 |
| count_085 | 18 | 14 | 18 | extreme | touching_pairs=4; distractors=32; occluders=4; texture_strength=15 |
| count_086 | 19 | 15 | 19 | extreme | touching_pairs=5; distractors=28; occluders=3; texture_strength=15 |
| count_087 | 20 | 14 | 20 | extreme | touching_pairs=6; distractors=29; occluders=3; texture_strength=15 |
| count_088 | 21 | 15 | 20 | extreme | touching_pairs=7; distractors=30; occluders=3; texture_strength=15 |
| count_089 | 22 | 17 | 22 | extreme | touching_pairs=4; distractors=31; occluders=3; texture_strength=15 |
| count_090 | 23 | 18 | 23 | extreme | touching_pairs=5; distractors=32; occluders=3; texture_strength=15 |
| count_091 | 24 | 19 | 24 | extreme | touching_pairs=6; distractors=28; occluders=4; texture_strength=15 |
| count_092 | 25 | 18 | 25 | extreme | touching_pairs=7; distractors=29; occluders=3; texture_strength=15 |
| count_093 | 25 | 23 | 25 | extreme | touching_pairs=4; distractors=30; occluders=3; texture_strength=15 |
| count_094 | 25 | 20 | 25 | extreme | touching_pairs=5; distractors=31; occluders=3; texture_strength=15 |
| count_095 | 25 | 19 | 25 | extreme | touching_pairs=6; distractors=32; occluders=3; texture_strength=15 |
| count_096 | 25 | 18 | 25 | extreme | touching_pairs=7; distractors=28; occluders=3; texture_strength=15 |
