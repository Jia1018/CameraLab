# Kubric Batch v2 Generation Plan

This document records the current full-dataset plan for paired Kubric videos.
It is meant to preserve the design decisions across container restarts and
future review sessions.

## Current Status

As of 2026-07-09, full large-scale data production has not started yet. The
completed work is still review/pilot work: `kubric_batch_v2_review_0000` is the
small pair-centric review run, `kubric_shape_speed_review_v1` is the focused
shape/speed review run, `kubric_object_motion_review_v1` is the broader
multi-object motion review, and `kubric_dynamic_shape_review_v1` is the targeted
non-box/non-sphere dynamic-shape review. None of these runs is the
training-scale batch.

The latest pre-production reviews expand object motion coverage to 2/3/4 dynamic
object templates, keep explicit static/baseline/brisk camera speed bands, change
the capsule shape check from an upright prop to a horizontal floor capsule, and
separately audit dynamic cylinder/capsule behavior. The next step before
production is human inspection of these review runs. If they pass, start the
first real pilot batch at 100-500 pair groups. Only after the pilot batch passes
coverage and quality checks should we launch the first training batch at 1k-5k
pair groups.

## Goal

Generate paired synthetic videos for learning to disentangle camera motion from
object/physical motion. The dataset should cover a broad range of plausible
camera trajectories, object dynamics, object appearances, materials, and scene
conditions while keeping exact pair contracts.

The implementation target is the official Kubric pipeline:

- `kubric.simulator.PyBullet` for rigid-body physics;
- `kubric.renderer.Blender` for rendering;
- per-clip metadata containing camera frames, physics frames, sampled
  parameters, quality audit results, and pair membership.

## Pair Contracts

Every pair has two clips. The two clips inside one pair must have the same FPS,
resolution, and frame count. Different pairs may have different frame counts.

Same-camera / different-physics pair:

- identical camera trajectory, including start position, start rotation/look-at,
  roll, lens, per-frame speed curve, FPS, and frame count;
- identical scene/background and lighting seed;
- different physics program and object parameters.

Same-physics / different-camera pair:

- identical physics simulation, object identities, object start states, object
  motion, collisions, scene/background, lighting, FPS, and frame count;
- different camera trajectory, including possible different start viewpoint,
  speed curve, roll, lens, and path family.

The relative initial composition is controlled only when required by a pair
contract. It is not globally fixed across the dataset, because object motion and
camera motion naturally change composition.

## Sampling Strategy

Use pair-centric sampling instead of rendering a full camera-family by
physics-family Cartesian grid. Each pair first chooses its controlled factor and
then samples one or two complementary variants. This keeps the review page small
while allowing the full batch to cover more kinds of camera and object motion.

Most continuous parameters should be sampled from clipped Gaussian
distributions, not from a few fixed values. The mean represents ordinary cases,
the standard deviation gives natural variation, and clipping prevents unusable
extremes. Important sampled values are written to metadata so later training and
filtering can condition on them.

The current v2 review generator follows this pattern in
`scripts/generate_kubric_batch_v2.py`.

## Camera Diversity

Camera families to cover:

- static off-center views;
- dolly in and dolly out;
- truck with counter-pan;
- crane/pedestal motion with tilt;
- top-down or ceiling-like views with drift;
- orbit arcs around the action;
- low-angle truck with mild roll;
- diagonal compound moves;
- occasional faster roll or wobble cases, kept rare because rapid camera
  rotation is uncomfortable and less common in real footage.

Sampled camera axes:

- start position: left, right, center, low, high, and top-down starts;
- look-at target: object-centered, slightly off-center, and drifting targets;
- translation magnitude and direction;
- average linear speed class: none, slow, medium, fast;
- speed multiplier band: most moving cameras use the baseline distribution,
  while a minority of samples use a `brisk` band capped at about 1.5x the
  baseline motion scale;
- speed curve inside a single clip: linear, ease-in, ease-out, ease-in-out, and
  two-stage curves, so a trajectory can accelerate or decelerate within the
  video;
- roll start and roll delta, usually small, with rare larger examples;
- focal length and mild zoom;
- path model: linear path or orbit path.

The goal is not to make every camera move dramatic. Most samples should be
human-viewable and moderate, with a smaller tail of faster or unusual motion.
The current generator records `speed_sampling_band` and `speed_multiplier` in
each camera spec so the faster tail can be audited instead of guessed from the
rendered video.

## Physics And Object Diversity

Physics families to cover:

- static or near-static object cases;
- gravity drop and bounce;
- single object moving toward/away from the camera;
- two-object collisions;
- sphere hitting a visible block or obstacle;
- box-sphere collision;
- three/four body scatter;
- chained multi-body collisions and crossfire-style multi-direction collisions;
- single-object and multi-object falling/bouncing, including staggered drops,
  simultaneous drops, and drop-then-collide cases;
- rolling, sliding, angular motion, and mixed translational/angular cases;
- out-of-frame or re-entering motion as a minority case.

Sampled physical axes:

- object count: start with 1-4 objects; expand later to 5+ after audits are
  stable;
- shape: sphere, cube/box, cylinder, cone, capsule, and imported Kubric/Blender
  assets when available;
- size: clipped Gaussian radius/extents per object family;
- mass/density: sampled with realistic bounds and correlated with size when
  possible;
- initial position and orientation;
- initial linear velocity and angular velocity;
- restitution, friction, damping, and material profile;
- color and surface appearance: matte, rough, smooth, glossy, metallic-like, or
  mixed;
- collision scenario: direct hit, glancing hit, miss/near miss only when
  intentionally labeled, bounce, roll, scatter, drop, rebound, and settle;
- gravity timing: single drops, multiple simultaneous drops, staggered drops,
  and falling objects that either collide with other objects or remain isolated.

Current implementation status: the official Kubric generator supports sphere,
cube/box, and procedural cylinder/cone/capsule bodies. The procedural shapes are
generated on demand as OBJ/URDF assets and rendered through Kubric/Blender rather
than committed as large mesh files. Procedural shapes can be generated along a
chosen principal axis, so capsule examples can be horizontal and floor-contacting
instead of upright props. Dynamic cylinder rolling, dynamic horizontal capsule
rolling, and sphere-to-dynamic-cylinder contact pass the current no-render audit.
The candidate dynamic cone slide failed ground-penetration audit and is excluded
until tuned separately.

Object and physics diversity should not come from invisible obstacles. If an
object changes motion due to a collision, the colliding object or surface should
be visible unless the scenario is explicitly labeled as an occlusion or
out-of-frame case.

Multi-object cases should not rely on one repeated role template. The current
plan uses multiple templates such as four-body scatter, three-body chain
collision, and four-body crossfire.  Within `same_physics` pairs, object colors,
materials, positions, and motion are intentionally identical because only the
camera is allowed to change. Across different physics samples, those values
should vary and be recorded in metadata.


## Complete Diversity Strategy

The dataset should be expanded through stratified families rather than a single
large undifferentiated random sampler. Each family has explicit metadata labels,
expected contacts when applicable, and a no-render audit before it is allowed
into rendered batches.

Camera diversity is already treated as a stratified family set: static,
dolly-in/out, truck/pan, crane/tilt, top-down drift, orbit, low truck with mild
roll, and diagonal compound motion. Within each camera family, start position,
look-at target, roll, lens, endpoint displacement, average speed band, and
within-clip speed curve are sampled from clipped Gaussian templates. Most camera
motion should remain human-viewable, with a smaller brisk tail capped around the
current 1.5x baseline scale.

Object and physics diversity should be covered by event families:

- static or near-static object arrangements for camera-only control cases;
- single-object gravity, bounce, roll, slide, and angular-motion cases;
- two-object collisions on or near the ground, including direct and glancing
  contacts;
- three/four-object scatter, chain, and crossfire interactions;
- multi-object gravity cases with simultaneous drops, staggered drops, and
  falling objects that either remain independent or collide;
- airborne collision cases where at least one pair contacts before ground
  contact;
- mixed-shape dynamic collisions involving sphere, box, cylinder, capsule, cone
  after each shape-specific audit passes;
- rare labeled near-miss or out-of-frame cases, kept separate from true-contact
  training pairs.

The airborne collision family should not stay as only one sphere-sphere example.
The intended variants are:

- sphere-sphere airborne contact, currently implemented as
  `phys_airborne_drop_collision`;
- sphere-box airborne contact;
- box-box or box-rect airborne glancing contact;
- sphere-cylinder airborne contact with visible post-impact spin; capsule
  airborne contact remains a future variant after its shape-specific audit;
- sequential airborne/ground chains where A hits B in air, then B or A hits C
  before or after first ground contact;
- multi-object mixed drops where 1-2 contact pairs are guaranteed by audit and
  other objects may fall independently.

The airborne collision review runs are intentionally focused stress tests; they
should not be read as the full dataset distribution. The broader pool also
contains ground collisions, rolling/sliding cases, static or near-static controls,
and a randomized mixed-drop family. `phys_randomized_mixed_drop_scene` samples
each object state independently from airborne, ground-moving, dynamic resting,
and static-ground modes. Object count, shape, size, XY position, height above
ground, linear velocity, angular velocity, material, friction, and restitution
are sampled per object, with only loose in-frame and initial non-overlap
constraints. Contacts are allowed but not forced. A `drop_timing_audit` rejects
samples where falling objects land too synchronously, currently requiring at
least a 14-frame spread among observed ground-contact times.

Continuous physical parameters should continue to use clipped Gaussian sampling:
object dimensions, masses, initial positions, linear velocities, angular
velocities, friction, restitution, damping, material profiles, and color. The
sampler should record these values in metadata so later analysis can audit the
actual generated distribution.

Quality gates before rendering:

- expected contact checks for labeled contact pairs;
- `expected_airborne_contacts` for pairs that must collide before ground contact;
- `drop_timing_audit` for randomized multi-object drops so independently falling
  objects do not collapse back into near-synchronous parallel drops;
- finite-motion, penetration, floating-rebound, sudden-stop, and bounce-complete
  plausibility audits;
- exact same-camera and same-physics pair-contract checks from metadata;
- review subsets rendered and inspected before scaling any new family.

Execution order:

1. Expand airborne collision templates from the current sphere-sphere case to
   sphere-box, box-box, sphere-cylinder/capsule, and sequential multi-object
   variants.
2. Run focused no-render audits for each new template, then a full-pool
   no-render audit covering every camera and physics family.
3. Render a compact review run, roughly 24-40 pair groups, selected for coverage
   rather than all camera-by-object combinations.
4. If the review looks physically plausible, start the next pilot batch at
   100-250 pair groups outside git and export a 24-40 pair GitHub Pages review
   sample.
5. After that pilot passes, scale to the planned 100-500 pair pilot range, then
   to the first 1k-5k pair training batch.

## Scene And Background Diversity

Scene variation should support motion understanding without adding confusing
static clutter.

Sampled scene axes:

- floor and wall color;
- subtle texture/material changes;
- wall height and side-wall configuration;
- lighting intensity, direction, and softness;
- occasional simple static background geometry.

Static background objects should be restrained. They are useful for camera
motion cues, but too many static shapes can be mistaken for physical actors.
When static background geometry exists, it must not overlap spawned dynamic
objects.

## Duration And Resolution

Pair members must share length. Different pairs can use different lengths.

Current practical defaults:

- FPS: 24;
- review resolution: 640x480;
- review duration range: roughly 3-6 seconds;
- large batch duration range: sample from a clipped Gaussian, with short,
  medium, and longer examples.

Physics speed and clip duration are separate variables. A longer video should
not automatically make object motion slower. Camera speed, object speed, and
duration should each be sampled and recorded independently, subject to quality
checks.

## Quality Audits

Before spending render time, each planned clip or pair should pass simulation
audits:

- expected contacts occur for collision-labeled scenarios;
- no severe object-object or object-ground penetration;
- no dynamic object starts overlapped with another object or confusing static
  geometry;
- bounce and settle behavior is physically plausible;
- no sudden stop without contact or clear damping explanation;
- finite object motion and finite camera transforms for all frames;
- same-camera and same-physics pair contracts compare exactly against metadata.

Rendered review audits:

- MP4 exists and has expected frame count;
- preview contact sheets are generated;
- rendered video is not noisy, blocky, or too compressed for inspection;
- visible actors match the metadata and scenario labels.

## Review And Publishing Policy

Small review runs are published to GitHub Pages through `docs/`. They should
show broad coverage, not every possible camera-object combination. A review run
should therefore include random or stratified pair combinations that expose many
camera families, physics families, object shapes, object counts, and
material/color profiles.

Large batch outputs should stay outside git, for example under
`/workspace/writeable/datasets/camera_motion_disentangle/<batch-id>`.

Use `scripts/export_kubric_review_sample.py` to publish sampled review subsets
from a large batch. The exporter keeps pair groups intact, only uses clips whose
videos and metadata are complete, greedily covers camera families, physics
families, pair kinds, speed classes, and duration bins, and copies only MP4s plus
metadata into `site/assets/runs/<review-id>`. Then run
`scripts/make_run_previews.py` and `scripts/sync_site_to_docs.sh` before pushing.

The site sync script copies `site/` to `docs/` and excludes raw `frames/`
directories. GitHub Pages should receive only MP4s, previews, manifests,
metadata, and summaries needed for review.

## Progress And Resume

Every batch run should write progress metadata while running:

- `progress.json` for current counts, frame totals, video totals, and status;
- `render_jobs.json` for planned render jobs;
- per-clip metadata before rendering;
- logs for simulation/render failures.

Use `scripts/update_kubric_progress.py` to refresh progress and
`scripts/resume_kubric_run.py` to continue a partially completed run after a
machine shutdown. This is required because the current machines can disappear
mid-run.

## Proposed Scale

Stage 1: review bank.

- 12-30 pair groups;
- 24-60 clips;
- 640x480;
- published to GitHub Pages for manual review.

Stage 1b: distribution-fix pilot.

- no-render audit for the updated camera-speed and multi-object templates;
- small rendered review subset after audits pass;
- check camera-speed histogram, multi-object color/material variety, and
  collision plausibility before production.

Stage 2: pilot batch.

- 100-500 pair groups;
- 200-1000 clips;
- run no-render audits first;
- render only after audits pass;
- inspect coverage histograms and a sampled web review subset.
- store the main pilot outside git and publish only a sampled review run, for
  example:
  `scripts/export_kubric_review_sample.py --source-run-dir /workspace/writeable/datasets/camera_motion_disentangle/<batch-id> --dest-run-id <batch-id>_review_sample --pairs 24 --overwrite`.

Stage 3: first training batch.

- 1k-5k pair groups;
- 2k-10k clips;
- store outside git;
- publish only a small curated review subset to `docs/`.

Stage 4: expanded dataset.

- 10k+ pair groups if storage and render budget allow;
- add more asset categories, material families, object counts, and scene
  layouts after the pilot confirms physical plausibility.

## Storage Expectation

Storage depends heavily on resolution, duration, codec settings, and whether raw
frames are kept.

Rules of thumb:

- GitHub Pages review subsets should stay small and exclude raw frames;
- raw rendered frames are much larger than MP4s and should be treated as
  temporary unless needed for debugging;
- keep large training batches outside the repo;
- record exact clip counts, durations, and average MP4 size after each pilot
  before scaling.

The v2 review run `kubric_batch_v2_review_0000` is the broad small published
example of this plan: 12 pair groups, 24 clips, 640x480, variable pair lengths,
official Kubric/PyBullet/Blender, progress tracking, and pair-centric coverage.
The follow-up run `kubric_shape_speed_review_v1` is a focused lightweight review:
6 clips, 320x240, 48 frames, procedural cylinder/cone/capsule coverage, and
static/baseline/brisk camera speed comparison. The review run
`kubric_object_motion_review_v1` has 18 clips, 12 pair groups, 640x480, 72
frames, static/baseline/brisk camera coverage, and six physics templates covering
shape checks plus 2/3/4 dynamic-object collisions. The targeted run
`kubric_dynamic_shape_review_v1` has 6 clips, 640x480, 72 frames, and covers
rolling dynamic cylinders, rolling horizontal capsules, and sphere contact with a
dynamic cylinder; dynamic cone was audited but excluded after penetration
failure.

The pilot sampling pool was first no-render audited across 26 pair groups / 52
clips, covering all 9 camera families and the earlier 13 physics families. A
focused airborne-collision pool audit then covered 18 physics families after
adding `phys_airborne_drop_collision`, `phys_airborne_sphere_box_collision`,
`phys_airborne_box_box_collision`, `phys_airborne_sphere_cylinder_collision`,
and `phys_airborne_chain_collision`. The current full-pool audit covers 19
physics families after adding `phys_randomized_mixed_drop_scene`: 38 pair groups
/ 76 clips, all 9 camera families, all 19 physics families, same-camera and
same-physics pairs balanced 19/19, and variable durations from 2.96s to 6.96s.
The focused review `kubric_random_mixed_drop_review_v2` has 4 pair groups / 8
clips and is intended specifically to inspect independent per-object
initialization in mixed drop scenes with explicit ground-object context.
