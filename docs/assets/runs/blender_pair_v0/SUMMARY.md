# blender_pair_v0

Generator: `blender_preview`

## Pair Groups

- `same_camera_diff_physics`: controlled `camera_id`, varied `physics_id`
  clips: A_cam_orbit_phys_bounce, B_cam_orbit_phys_roll
- `diff_camera_same_physics`: controlled `physics_id`, varied `camera_id`
  clips: A_cam_orbit_phys_bounce, C_cam_dolly_phys_bounce

## Clips

- `A_cam_orbit_phys_bounce`: camera `cam_orbit_left`, physics `phys_bounce`
  preview: `previews/A_cam_orbit_phys_bounce_sheet.png`
  video: `videos/A_cam_orbit_phys_bounce.mp4`
  metadata: `metadata/A_cam_orbit_phys_bounce.json`
- `B_cam_orbit_phys_roll`: camera `cam_orbit_left`, physics `phys_roll`
  preview: `previews/B_cam_orbit_phys_roll_sheet.png`
  video: `videos/B_cam_orbit_phys_roll.mp4`
  metadata: `metadata/B_cam_orbit_phys_roll.json`
- `C_cam_dolly_phys_bounce`: camera `cam_dolly_in`, physics `phys_bounce`
  preview: `previews/C_cam_dolly_phys_bounce_sheet.png`
  video: `videos/C_cam_dolly_phys_bounce.mp4`
  metadata: `metadata/C_cam_dolly_phys_bounce.json`
