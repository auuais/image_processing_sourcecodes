# 5-6 Page Presentation Outline

## Page 1. Problem definition

- Title: `Multi-Label Packaging Defect Inspection Using Feature-Based Alignment`
- Goal: inspect different consumer-package fronts and decide whether each sample is normal or defective
- Industrial relevance:
  - packaging quality control
  - automatic rejection of defective products before shipment

## Page 2. Dataset

- 5 package labels from Open Food Facts:
  - cookies
  - cereal
  - chips
  - tea
  - chocolate
- 4 inspection samples per label:
  - 2 PASS
  - 2 FAIL
- Defects used:
  - missing print
  - scratch
  - stain
  - corner damage

## Page 3. Method

- Step 1: load the correct reference label for the sample
- Step 2: detect SIFT keypoints on the reference and sample image
- Step 3: match features and estimate homography
- Step 4: align the sample to the reference
- Step 5: compute absolute difference after alignment
- Step 6: use thresholding, morphology, and connected components
- Step 7: decide PASS or FAIL from defect area ratio

## Page 4. Representative results

- Show one PASS sample from one label
- Show two FAIL samples from different labels
- Explain why PASS leaves almost no residual area and FAIL leaves a large connected region

## Page 5. Quantitative summary

- Report:
  - number of labels
  - total samples
  - accuracy
  - average inlier count
- Show per-sample defect area chart
- Show mean PASS vs FAIL defect area by label

## Page 6. Discussion

- Strengths:
  - more realistic than one synthetic package
  - explainable output
  - direct use of course topics
- Limitations:
  - defects are still simulated
  - real factory lighting and blur are not yet included
- Future work:
  - collect real conveyor or phone images
  - compare SIFT with ORB / AKAZE
  - add illumination normalization
