# ⭐️⭐️MVSE-3D Model Code Help 

## Introduction

#### This is the official code of MFIANet: Dense matching algorithms face challenges in disparity discontinuity regions and weak/repetitive texture areas, leading to surface irregularities, geometric detail loss, and void artifacts in oblique photogrammetric 3D models. To address these limitations, this study proposes a novel geometry optimization framework guided by image-space semantic edge features through geometric mapping between 2D imagery and 3D models. Our methodology initiates with extracting structural point clouds from building models through geometric-topological analysis and multi-view texture coherence assessment. A hybrid feature extraction strategy combines Line Segment Detector (LSD)-based structural line detection with deep learning-enhanced edge characterization across multi-perspective imagery. The framework implements a multi-stage refinement process: 1) Depth-First Search (DFS)-based edge component labeling with multi-view geometric constraints, 2) Non-structural component rejection through attribute-driven filtering and projective geometry analysis, 3) Adaptive dynamic buffer zones with Markov Random Field (MRF) optimization for optimal structural line association, and 4) Epipolar-constrained position refinement using multi-homography geometric verification. Experimental validation demonstrates our method's superiority in architectural point cloud localization accuracy, achieving 20.43% average reduction in point-to-mesh distance compared to conventional approaches, and showing significant improvements in surface continuity and void reduction. The framework mitigates local deformations while preserving architectural details, establishing a robust solution for photogrammetric 3D model optimization.

## File Directory

| Filename                  | Brief                                                        |
| ------------------------- | ------------------------------------------------------------ |
| GGLI.py                   | GGLI Vegetation Index Calculation Code                       |
| curvature.py              | Building point cloud curvature constraint code               |
| point cloud constraint.py | Point cloud projection, DFS-based connected component search and labeling, dynamic buffer construction, point-line association and epipolar constraint code |

## Dataset

#### The three datasets used in this paper are: ECUT-SMB (a self-built dataset), Dortmund, and Zeche (public datasets).

| dataset  | URL                                                      |
| -------- | -------------------------------------------------------- |
| ECUT-SMB | https://pan.baidu.com/s/1tvsE9RObJHkK2UWQ-woHZw?pwd=5zsp |
| Dortmund | https://eostore.itc.utwente.nl:5001/sharing/3Jhp6O628    |
| Zeche    | https://eostore.itc.utwente.nl:5001/sharing/f49J04Pjc    |
