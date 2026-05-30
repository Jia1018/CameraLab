# blender_rich_v0

Generator: `blender_rich_preview`

## Pair Groups

- `same_camera_combo_diff_physics`: controlled `camera_id`, varied `physics_id`
  clips: A_combo_bounce_diag, B_combo_two_cross
- `different_camera_same_multibody_physics`: controlled `physics_id`, varied `camera_id`
  clips: B_combo_two_cross, C_orbit_two_cross, D_dolly_two_cross
- `same_truck_camera_single_vs_multi`: controlled `camera_id`, varied `physics_id`
  clips: E_truck_roll_reverse, F_truck_multi_swirl

## Clips

- `A_combo_bounce_diag`: camera `cam_combo_crane_orbit`, physics `phys_bounce_diag`
  preview: `previews/A_combo_bounce_diag_sheet.png`
  video: `videos/A_combo_bounce_diag.mp4`
  metadata: `metadata/A_combo_bounce_diag.json`
- `B_combo_two_cross`: camera `cam_combo_crane_orbit`, physics `phys_two_cross`
  preview: `previews/B_combo_two_cross_sheet.png`
  video: `videos/B_combo_two_cross.mp4`
  metadata: `metadata/B_combo_two_cross.json`
- `C_orbit_two_cross`: camera `cam_orbit_fast`, physics `phys_two_cross`
  preview: `previews/C_orbit_two_cross_sheet.png`
  video: `videos/C_orbit_two_cross.mp4`
  metadata: `metadata/C_orbit_two_cross.json`
- `D_dolly_two_cross`: camera `cam_dolly_in_tilt`, physics `phys_two_cross`
  preview: `previews/D_dolly_two_cross_sheet.png`
  video: `videos/D_dolly_two_cross.mp4`
  metadata: `metadata/D_dolly_two_cross.json`
- `E_truck_roll_reverse`: camera `cam_truck_right`, physics `phys_roll_reverse`
  preview: `previews/E_truck_roll_reverse_sheet.png`
  video: `videos/E_truck_roll_reverse.mp4`
  metadata: `metadata/E_truck_roll_reverse.json`
- `F_truck_multi_swirl`: camera `cam_truck_right`, physics `phys_multi_swirl`
  preview: `previews/F_truck_multi_swirl_sheet.png`
  video: `videos/F_truck_multi_swirl.mp4`
  metadata: `metadata/F_truck_multi_swirl.json`
