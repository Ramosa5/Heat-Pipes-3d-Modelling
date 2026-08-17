# Modular Bubble 3D Reconstruction

This is the modularized version of the original single-file script.

`main.py` is only a hub: it parses options, builds `ReconstructionConfig`, and calls the pipeline orchestrator. The actual work is distributed into modules inside `bubble_reconstruction/`.

## Structure

```text
main.py                              # entry point / orchestration hub
requirements.txt
bubble_reconstruction/
  config.py                          # central settings and default tube geometry
  processing.py                      # high-level pipeline orchestration
  coco_utils.py                      # COCO loading and annotation-to-mask conversion
  tube_geometry.py                   # tube membership and mask overlay helpers
  rectification.py                   # crop, perspective transform, alignment
  components.py                      # connected components and longitudinal matching
  volume.py                          # volume reconstruction and point conversion
  reconstruction.py                  # per-bubble no-stick reconstruction
  fit_score.py                       # rotational fit score and validation
  eccentricity.py                    # bubble tip eccentricity from reconstructed 3D points
  bubble_physics.py                  # mask-height parameters from rectified masks
  parameter_export.py                # CSV export for calculated frame parameters
  pipe.py                            # PyVista pipe/cylinder mesh
  export_io.py                       # mask and PLY saving
  visualization.py                   # PyVista live animation and camera setup
legacy_sources/
  20260531_parameters.py             # original uploaded source file
  20260301_bubble_physics.py         # original uploaded source file
  tempCodeRunnerFile.py              # original scratch/duplicate source file
```

## Install

```bash
pip install -r requirements.txt
```

## Run like the previous script

```bash
python main.py
```

Default behavior:

- `start_frame=100`
- `n_frames=10`
- `save_masks=False`
- `save_point_clouds=False`
- preview enabled
- eccentricity/parameter export disabled

## Useful commands

Run without the PyVista preview window:

```bash
python main.py --no-preview
```

Save masks and point clouds:

```bash
python main.py --save-masks --save-point-clouds
```

Choose dataset and frame range:

```bash
python main.py --dataset-dir bubble.coco/train --start-frame 1 --n-frames 40
```

Validate the rotational fit score using synthetic data:

```bash
python main.py --validate-fit-score
```

## Eccentricity and frame-parameter export

Enable the new frame-by-frame parameter calculation with:

```bash
python main.py --eccentricity --no-preview
```

This creates CSV files in the `parameters/` folder:

```text
parameters/eccentricity_parameters.csv
parameters/mask_parameters.csv
```

The eccentricity CSV contains one row per reconstructed bubble in each tube pair:

```text
frame_no,file_name,tube_pair,bubble_index,bubble_count,e_star,e_x_mm,e_y_mm,tip_percentile,diameter_mm
```

The mask-parameter CSV contains one row per rectified view in each tube pair:

```text
frame_no,file_name,tube_pair,view,diameter_mm,mm_per_pixel,columns_total,columns_nonzero,avg_height_mm,max_height_mm,avg_alpha_filled,avg_alpha_empty,avg_s_filled_mm,avg_s_interface_mm
```

Custom output folder and tip percentile:

```bash
python main.py --eccentricity --parameters-dir parameters --tip-percentile 99.0 --no-preview
```

Detailed eccentricity diagnostics:

```bash
python main.py --eccentricity --eccentricity-debug --no-preview
```

### Eccentricity visualization

The original visualization style from `20260531_parameters.py` is now available from the active pipeline:

```bash
python main.py --eccentricity --eccentricity-visualize
```

This opens a PyVista window for each visualized frame and tube pair:

- full reconstructed bubble cloud: light gray,
- selected tip slice used for eccentricity: red,
- median eccentricity point: blue sphere.

For long sequences, avoid opening a window for every frame:

```bash
python main.py --eccentricity --eccentricity-visualize --eccentricity-visualize-every 10
```

`--eccentricity-visualize` automatically enables eccentricity calculation, so it still writes the CSV files.

## Bubble tracking

Enable frame-to-frame bubble tracking with:

```bash
python main.py --tracking --no-preview
```

This creates:

```text
parameters/tracking_parameters.csv
```

The tracker works after the existing per-bubble TOP/SIDE reconstruction. For every frame and tube pair, it:

1. reconstructs each matched bubble as an individual 3D volume,
2. calculates the 3D centroid and axial span of every detected bubble,
3. matches detections to active tracks using nearest 3D centroid distance,
4. gives new bubbles a new persistent `track_id`,
5. closes tracks when the bubble disappears.

Default behaviour closes a track immediately when the bubble is not detected in the next processed frame:

```bash
python main.py --tracking --tracking-max-missing-frames 0 --no-preview
```

You can allow short gaps, for example one missed frame:

```bash
python main.py --tracking --tracking-max-missing-frames 1 --no-preview
```

The maximum matching distance is configured in millimetres:

```bash
python main.py --tracking --tracking-max-distance-mm 12 --no-preview
```

CSV columns:

```text
frame_no,file_name,tube_pair,track_id,detection_index,match_status,match_distance_mm,track_age_frames,missed_frames,centroid_x_mm,centroid_y_mm,centroid_z_mm,z_min_mm,z_max_mm,volume_voxels,point_count
```

To show the track IDs in the PyVista live preview:

```bash
python main.py --tracking-labels
```

`--tracking-labels` automatically enables tracking. The CSV is still written.

## 2D frame parameter overlay

Save original frames with the calculated per-bubble values drawn directly next to the detected bubbles:

```bash
python main.py --annotate-frame-parameters --no-preview
```

This creates PNG files in:

```text
annotated_frames/
```

`--annotate-frame-parameters` automatically enables both tracking and eccentricity calculation, because the overlay needs persistent IDs and eccentricity values. The frame overlay contains:

- original COCO bubble annotation boxes and contours,
- tube reference lines,
- persistent tracking ID,
- detection index,
- reconstructed longitudinal centroid `z`,
- reconstructed volume in voxels `V`,
- eccentricity parameters `e*`, `ex`, and `ey`.

Example with a custom output folder:

```bash
python main.py --annotate-frame-parameters --annotated-frames-dir annotated_frames --no-preview
```

You can combine it with normal CSV export and point-cloud export:

```bash
python main.py --tracking --eccentricity --annotate-frame-parameters --save-point-clouds --no-preview
```


## 3D parameter labels

Use `python main.py --preview-parameter-labels` to show per-bubble labels directly in the PyVista 3D preview. This automatically enables tracking and eccentricity so each bubble label can include: track ID, detection index, centroid Z, volume in voxels, `e*`, `e_x`, and `e_y`.


## Per-bubble rotational-fit parameters

Use `python main.py --rotational-fit-parameters --no-preview` to calculate per-bubble rotational-fit values from each individual reconstructed bubble. The output is saved to `parameters/rotational_fit_parameters.csv`. The values include `rotational_fit_score` (`I_rot`), mean radial error, and fitted reference radius.

Use `python main.py --preview-parameter-labels` to show these rotational-fit values directly in the 3D PyVista labels together with track ID and eccentricity values. This flag automatically enables tracking, eccentricity, and rotational-fit parameters.

## Front/back eccentricity parameter

Use `python main.py --front-back-eccentricity --no-preview` to calculate the front/tip and back/tail eccentricity of each reconstructed bubble. The output is saved to `parameters/front_back_eccentricity_parameters.csv`.

Use `python main.py --preview-parameter-labels` to show these values directly inside the PyVista 3D preview together with the other per-bubble labels. The rich label includes `e_front`, `e_back`, front/back transverse shifts, clipping flags, tracking ID, standard eccentricity, and rotational-fit parameters.


### Full parameter names in 3D labels

The `--preview-parameter-labels` view displays full parameter names together with abbreviations, for example `Tracking ID (ID)`, `Tip eccentricity (e*)`, `Rotational fit index (I_rot)`, and `Front eccentricity (e_front)`.

## Fullscreen / compact preview

Use fullscreen startup for the PyVista 3D preview:

```bash
python main.py --preview-parameter-labels --preview-fullscreen
```

Use a more minimal preview with smaller labels, smaller points and a slightly more zoomed-out camera:

```bash
python main.py --preview-parameter-labels --preview-fullscreen --preview-compact
```

When fullscreen is not used, the preview window size can be selected explicitly:

```bash
python main.py --preview-parameter-labels --preview-window-size 1800 1000
```


## Minimized / distant 3D preview

If the pipes still appear too close after opening the PyVista window, use the distance scale option. It starts the camera farther away and applies an additional scroll-wheel-like zoom out:

```bash
python main.py --preview-parameter-labels --preview-fullscreen --preview-compact --preview-distance-scale 5
```

Increase the value, for example `8` or `10`, when the pipes should start even smaller.


## End-of-processing summary visualization

Use `python main.py --summary-visualization --no-preview` to open a single smooth summary window after processing. The window shows the original video frame on top, the reconstructed pipes in the middle, and two time-series plots at the bottom: `e(t)` from eccentricity (`e*`) and `eta(t)` from the rotational-fit index (`I_rot`) for all tracked bubbles.

## Video source mode

The normal COCO workflow is still available:

```bash
py main.py --source coco --dataset-dir bubble.coco/train --coco-file _annotations.coco.json --start-frame 89 --n-frames 10 --summary-visualization --no-preview
```

The new video workflow uses Mask R-CNN predictions instead of COCO annotations:

```bash
py main.py --source video --video C001H002S0001.avi --model new_best_maskrcnn_bubble.pth --start-frame 89 --n-frames 10 --summary-visualization --no-preview
```

In video mode, Mask R-CNN masks are predicted for each selected video frame, assigned to tube1/tube2/tube3/tube4 by bbox centre, and then passed into the same rectification and 3D reconstruction pipeline used by COCO mode.

Optional debug video with 2D Mask R-CNN detections:

```bash
py main.py --source video --video C001H002S0001.avi --model new_best_maskrcnn_bubble.pth --start-frame 89 --n-frames 10 --summary-visualization --no-preview --save-detection-video detected_input.avi
```
