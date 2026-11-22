# Cassava YOLO Benchmark — Final Report

## Dataset and splits
- Dataset: iCassava 2019 / FGVCx, 5 classes: cbb, cbsd, cgm, cmd, healthy.
- Internal split (seed 42): 4512 train / 564 validation / 565 hold-out test images.
- Kaggle test set: 3774 unlabeled images; submissions are checked structurally only, never against hidden labels.

## Availability and routes
| Route | Models | Meaning | Status |
|---|---|---|---|
| R1 | v8 n/s, v11 n/s | native ImageNet-pretrained *-cls weights | 4 trained |
| R3 | v3 n, v5 n/s, v9 n/s, v10 n/s, v12 n/s | COCO detection backbone + new Classify head | 9 trained |
| Skipped | v4, v6, v7 | no usable pretrained classification/detection weights in scope | 0 trained |
- v3 is labelled **YOLOv3-tiny** everywhere in this report and in all generated tables/charts.
- results\failures.csv absent; no failed runs were recorded (all 13 passed).
- Duplicate-score check: v10_n_pretrained_seed0 = 457/564, v12_n_pretrained_seed0 = 457/564, v12_s_pretrained_seed0 = 457/564. The three best.pt files have distinct SHA-256 hashes and the three saved validation prediction CSVs have distinct SHA-256 hashes.

## Protocol
| Item | Setting |
|---|---|
| Dataset | iCassava 2019 / FGVCx, 5 classes |
| Internal split | 4512 train / 564 val / 565 hold-out test; split seed 42 |
| Kaggle test | 3774 unlabeled images; submissions structurally validated only |
| Training seed | 0 |
| Epochs / patience | 30 / 10 |
| Batch / image size | 16 / 224x224 |
| Optimizer / schedule | AdamW lr 0.001, cosine to 1% |
| Loss | class-weighted cross-entropy + label smoothing 0.1 |
| AMP / workers / cache | on / 2 / off |
| Hold-out use | opened once per final model after selection |

## Ranked validation comparison (all 13 models)
| Rank | Model | Route | Val accuracy | Val macro P | Val macro R | Val macro F1 |
|---|---|---|---|---|---|---|
| 1 | yolo11s-cls | R1 | 0.870567 | 0.810623 | 0.855079 | 0.829294 |
| 2 | yolov8s-cls | R1 | 0.868794 | 0.809924 | 0.855655 | 0.828997 |
| 3 | yolo11n-cls | R1 | 0.856383 | 0.780016 | 0.839377 | 0.803592 |
| 4 | YOLOv3-tiny | R3 | 0.854610 | 0.786222 | 0.825316 | 0.803331 |
| 5 | yolov8n-cls | R1 | 0.845745 | 0.777910 | 0.818939 | 0.794849 |
| 6 | yolov9s-cls | R3 | 0.842199 | 0.768255 | 0.816750 | 0.787018 |
| 7 | yolov9t-cls | R3 | 0.824468 | 0.747456 | 0.800906 | 0.767563 |
| 8 | yolov5n-cls | R3 | 0.820922 | 0.745819 | 0.803104 | 0.763419 |
| 9 | yolov5s-cls | R3 | 0.813830 | 0.733586 | 0.790702 | 0.755818 |
| 10 | yolo12n-cls | R3 | 0.810284 | 0.738163 | 0.779808 | 0.750708 |
| 11 | yolo12s-cls | R3 | 0.810284 | 0.730049 | 0.762876 | 0.743706 |
| 12 | yolov10n-cls | R3 | 0.810284 | 0.720998 | 0.762868 | 0.736680 |
| 13 | yolov10s-cls | R3 | 0.771277 | 0.698106 | 0.736655 | 0.708735 |
- Route-tagged table: `D:\personal projects\yolo models project\project\results\comparison_val.csv`
- One grouped bar chart: `D:\personal projects\yolo models project\project\results\comparison_val.png`

## Final hold-out results
| Model | Route | Images | Accuracy | Macro F1 | Balanced accuracy |
|---|---|---|---|---|---|
| v11s (final) | R1 | 565 | 0.861947 | 0.812358 | 0.830601 |
| v8s (runner-up) | R1 | 565 | 0.874336 | 0.829212 | 0.845607 |
- Final model was selected by validation macro-F1 before the hold-out was opened: `v11_s_pretrained_seed0` (0.829294), a near-tie with `v8_s_pretrained_seed0` (0.828997).
- The `v11_s_pretrained_seed0` hold-out evaluation was reused from its single prior evaluation; `v8_s_pretrained_seed0` was evaluated once in this run. No model was evaluated twice.

## Per-class results — final model v11s on hold-out
| Class/metric | Precision | Recall | F1 | One-vs-rest accuracy | Support | Value |
|---|---|---|---|---|---|---|
| cbb | 0.688889 | 0.659574 | 0.673913 | 0.946903 | 47 |  |
| cbsd | 0.869565 | 0.833333 | 0.851064 | 0.925664 | 144 |  |
| cgm | 0.790698 | 0.883117 | 0.834356 | 0.952212 | 77 |  |
| cmd | 0.934109 | 0.906015 | 0.919847 | 0.925664 | 266 |  |
| healthy | 0.710526 | 0.870968 | 0.782609 | 0.973451 | 31 |  |
| macro_avg | 0.798757 | 0.830601 | 0.812358 | 0.830601 | 565 |  |
| weighted_avg | 0.865448 | 0.861947 | 0.862677 | 0.830601 | 565 |  |
| overall_accuracy |  |  |  |  |  | 0.861947 |
| balanced_accuracy |  |  |  |  |  | 0.830601 |

## Per-class results — runner-up v8s on hold-out
| Class/metric | Precision | Recall | F1 | One-vs-rest accuracy | Support | Value |
|---|---|---|---|---|---|---|
| cbb | 0.761905 | 0.680851 | 0.719101 | 0.955752 | 47 |  |
| cbsd | 0.885714 | 0.861111 | 0.873239 | 0.936283 | 144 |  |
| cgm | 0.823529 | 0.909091 | 0.864198 | 0.961062 | 77 |  |
| cmd | 0.930502 | 0.906015 | 0.918095 | 0.923894 | 266 |  |
| healthy | 0.692308 | 0.870968 | 0.771429 | 0.971681 | 31 |  |
| macro_avg | 0.818792 | 0.845607 | 0.829212 | 0.845607 | 565 |  |
| weighted_avg | 0.877415 | 0.874336 | 0.874717 | 0.845607 | 565 |  |
| overall_accuracy |  |  |  |  |  | 0.874336 |
| balanced_accuracy |  |  |  |  |  | 0.845607 |

## Hold-out artifacts (full PNG paths)
### v11_s_pretrained_seed0 hold-out PNGs
- `D:\personal projects\yolo models project\project\results\final\v11_s_pretrained_seed0_test\confusion_matrix.png`
- `D:\personal projects\yolo models project\project\results\final\v11_s_pretrained_seed0_test\confusion_matrix_normalized.png`
- `D:\personal projects\yolo models project\project\results\final\v11_s_pretrained_seed0_test\overall_metrics.png`
- `D:\personal projects\yolo models project\project\results\final\v11_s_pretrained_seed0_test\per_class_accuracy.png`
- `D:\personal projects\yolo models project\project\results\final\v11_s_pretrained_seed0_test\per_class_prf.png`

### v8_s_pretrained_seed0 hold-out PNGs
- `D:\personal projects\yolo models project\project\results\final\v8_s_pretrained_seed0_test\confusion_matrix.png`
- `D:\personal projects\yolo models project\project\results\final\v8_s_pretrained_seed0_test\confusion_matrix_normalized.png`
- `D:\personal projects\yolo models project\project\results\final\v8_s_pretrained_seed0_test\overall_metrics.png`
- `D:\personal projects\yolo models project\project\results\final\v8_s_pretrained_seed0_test\per_class_accuracy.png`
- `D:\personal projects\yolo models project\project\results\final\v8_s_pretrained_seed0_test\per_class_prf.png`

Per-image predictions and per-class CSVs are in `results\final\<run>_test\`.

## Efficiency (all 13 models)
| Model | Route | Params (M) | GFLOPs @224 | best.pt (MB) | Latency b1 ms/img | Latency b32 ms/img |
|---|---|---|---|---|---|---|
| yolov8n-cls | R1 | 1.445 | 0.206 | 2.97 | 6.473 | 0.276 |
| yolov8s-cls | R1 | 5.087 | 0.771 | 10.27 | 6.076 | 0.595 |
| yolo11n-cls | R1 | 1.538 | 0.199 | 3.19 | 9.043 | 0.304 |
| yolo11s-cls | R1 | 5.449 | 0.743 | 11.03 | 8.278 | 0.636 |
| YOLOv3-tiny | R3 | 2.238 | 0.349 | 4.51 | 1.797 | 0.390 |
| yolov5n-cls | R3 | 1.383 | 0.190 | 2.87 | 8.568 | 0.320 |
| yolov5s-cls | R3 | 4.836 | 0.675 | 9.78 | 11.561 | 0.597 |
| yolov9t-cls | R3 | 0.881 | 0.182 | 2.02 | 19.090 | 0.650 |
| yolov9s-cls | R3 | 3.157 | 0.687 | 6.58 | 23.703 | 0.905 |
| yolov10n-cls | R3 | 1.536 | 0.211 | 3.19 | 11.179 | 0.447 |
| yolov10s-cls | R3 | 4.553 | 0.747 | 9.26 | 16.218 | 0.703 |
| yolo12n-cls | R3 | 1.725 | 0.223 | 3.63 | 25.021 | 0.862 |
| yolo12s-cls | R3 | 6.111 | 0.828 | 12.43 | 28.187 | 1.320 |
- Params: `sum(p.numel())` from each loaded checkpoint.
- GFLOPs: `thop.profile` with a 1x3x224x224 input.
- Latency: GPU inference on **NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM**, warmup 20 iterations, mean over 200 timed iterations, batch 1 and batch 32. These are laptop-GPU numbers, not datacenter-GPU numbers.

## Statistics
| Model | Accuracy | Accuracy 95% CI | Macro F1 | Macro F1 95% CI |
|---|---|---|---|---|
| v11_s_pretrained_seed0 | 0.861947 | 0.831858-0.886770 | 0.812358 | 0.768863-0.849177 |
| v8_s_pretrained_seed0 | 0.874336 | 0.846018-0.900885 | 0.829212 | 0.784458-0.868278 |
- Bootstrap: 1000 resamples, seed 0.
- McNemar (v11s vs v8s): v11s correct / v8s wrong = 21; v11s wrong / v8s correct = 28; discordant = 49; exact two-sided p = 0.391603.
- Plain language: the v11s-vs-v8s difference is **NOT statistically significant at alpha=0.05**.

## Error analysis
Top off-diagonal hold-out confusions for v11s:
| Count | True -> Predicted |
|---|---|
| 10 | cmd -> cgm |
| 10 | cmd -> cbsd |
| 10 | cbsd -> cbb |
| 7 | cgm -> cmd |
| 7 | cbsd -> cmd |
| 6 | cbb -> cbsd |
| 4 | cbsd -> cgm |
| 4 | cbb -> healthy |
- The main errors are between visually overlapping foliar symptom classes. In particular, cbsd and cmd both produce irregular brown/gray leaf lesions under field lighting; cbb and cgm can both present with yellowing and wilting. Class imbalance also matters: cmd has 266 hold-out images while healthy has 31, so minority-class F1 is weaker and errors concentrate in the larger lesion classes.

## Submissions
| File | Rows | Structurally valid |
|---|---|---|
| submission_v11_s_pretrained_seed0_plain.csv | 3774 | True |
| submission_v11_s_pretrained_seed0_flip_tta.csv | 3774 | True |
| submission_v8_s_pretrained_seed0_plain.csv | 3774 | True |
| submission_v8_s_pretrained_seed0_flip_tta.csv | 3774 | True |
| submission_v11_n_pretrained_seed0_plain.csv | 3774 | True |
| submission_v11_n_pretrained_seed0_flip_tta.csv | 3774 | True |
| submission_top3_mean_softmax.csv | 3774 | True |
- Top 3 by validation macro-F1: v11_s_pretrained_seed0, v8_s_pretrained_seed0, v11_n_pretrained_seed0.
- Variants: plain, horizontal-flip TTA, and one mean-softmax ensemble of the three top models' plain probabilities.
- All CSVs were validated against `D:\personal projects\yolo models project\project\data\sampleSubmission.csv`: columns `Category,Id`, 3774 rows, no duplicate or missing Ids, filenames with extension, and labels within `{cbb,cbsd,cgm,cmd,healthy}`. Hidden-label accuracy cannot be checked.

## Recommendations
- Use the top-3 mean-softmax ensemble if Kaggle leaderboard latency is not binding; otherwise v11s is the single best validation model.
- Increase R3 training beyond 30 epochs with early stopping, since R3 models were still improving late in training.
- Repeat with multiple seeds/splits and add external-field images to reduce sensitivity to one split.
- Consider class-balanced sampling or focal loss for cbb/healthy, and collect harder cbsd/cmd examples.

## Limitations
- Single seed and single split.
- n/s sizes only; no m/l/x models.
- Latency measured on a 4 GB laptop GPU, not a datacenter GPU.
- R3 models ran all 30 epochs and may be undertrained.
- R1 vs R3 pretraining differs: R1 uses ImageNet classifier weights, R3 uses COCO detector backbones with a new Classify head; differences are not purely architectural.
- Kaggle hidden labels were not available, so submissions were checked structurally only.

## Citation
Mwebaze et al., "iCassava 2019 Fine-Grained Visual Categorization Challenge", https://arxiv.org/pdf/1908.02900.pdf
Ultralytics YOLO: https://github.com/ultralytics/ultralytics
