# Lecture 12

This folder contains the lecture 12 practice code for the classical detection and recognition implementation slides at the end of `lecture12.pdf`.

## Files

- `common.py`: shared helpers, official sample downloads, and synthetic scene generation
- `page67_pedestrian_detection_using_hog_svm.py`: pedestrian detection with OpenCV's HOG + linear SVM detector
- `page68_optical_character_recognition_using_knn_and_svm.py`: OCR on the OpenCV digits sheet using KNN and SVM classifiers
- `page69_face_detection_using_haar_and_lbp_features.py`: frontal-face detection comparison using Haar and LBP cascades
- `page70_aruco_pattern_detector.py`: ArUco marker detection on a locally generated marker scene

## Local data

The scripts create their own local resources in `data/` when first run:

- `data/pedestrians`: official OpenCV basketball sample images
- `data/ocr`: the official `digits.png` OCR sample sheet
- `data/faces`: `lena.jpg`, the official LBP cascade XML, and a generated multi-face test scene
- `data/aruco`: a generated marker scene for ArUco detection

Sources:

- `opencv/samples/data` for `basketball1.png`, `basketball2.png`, `digits.png`, and `lena.jpg`
- `opencv/data/lbpcascades` for `lbpcascade_frontalface_improved.xml`

User-provided files placed in `data/new_data/` are preferred automatically when a matching lecture task can use them.

## How to run

Run any script from inside `lecture12`:

```powershell
python page67_pedestrian_detection_using_hog_svm.py
```

Optional flags:

- `--show` opens the Matplotlib figure after saving

## Output

- Figures are saved in `output/`
- Downloaded and generated resources are saved in `data/`

## Note

This lecture PDF does not include a separate assignment slide after the implementation section. The last implementation page, page 70, is treated as the end-of-lecture final task.
