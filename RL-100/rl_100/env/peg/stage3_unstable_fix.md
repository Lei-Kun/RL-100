# Stage3 Unstable Fix

## Summary

This note records the debugging process and the final runtime fix for the
juicing task instability.

The juicing task is executed in three stages:

1. Stage 1: pick the orange and place it into the juicer.
2. Stage 2: execute a fixed trajectory to press the lever.
3. Stage 3: pick the residue and place it into the trash.

The real safety issue was concentrated in Stage 3. The dominant failure mode was
not large TCP translation. It was aggressive wrist reorientation near the
residue, sometimes followed by IK / servo instability and rare "fly away"
behavior.


## Problem Characterization

### 1. Euler-angle branch switching

The policy outputs absolute `rpy`. During wrist turning, the same physical
orientation can be represented by multiple Euler tuples. This produced large
numerical jumps in the action stream and caused visible shaking.

Observed symptom:

- raw `rpy` could jump by more than 200 deg between adjacent steps while the
  physical rotation was still close.

Early false-positive examples:

- `episode_20260328_152733_juicing_all_stage1_ep001.jsonl`
- `episode_20260328_152802_juicing_all_stage1_ep001.jsonl`

These logs showed that directly faulting on Euler-angle differences incorrectly
killed normal wrist flips.

### 2. True dangerous Stage 3 wrist burst

After removing the Euler false positives, the remaining Stage 3 issue was still
real: the policy sometimes asked for a large wrist reorientation burst while TCP
position remained relatively stable.

Representative successful-but-risky Stage 3 log:

- `episode_20260328_154639_juicing_all_stage3_ep002.jsonl`

Important window in that log:

- steps `141-145`
- `cmd_rotation_distance_deg` peaked at about `89.46`
- `cmd_clip_rpy_norm_deg` peaked at about `69.46`
- `measured_delta_rpy_deg_unwrapped_norm` peaked at about `30.34`
- measured TCP position remained small, `max measured_delta_pos_norm_m` stayed
  around `0.00174`

Interpretation:

- the arm was not translating dangerously
- the wrist was rapidly trying to reorient for grasp alignment
- this matched the real robot observation: the arm looked like it was forcing
  the grasp orientation instead of approaching smoothly

### 3. Hard freeze solved safety but caused hesitation

The first Stage 3-specific fix introduced orientation freezing. This stopped the
dangerous burst but overconstrained the execution.

Representative log:

- `episode_20260328_155837_juicing_all_stage3_ep002.jsonl`

Important observation:

- Stage 3 entered `frozen` at step `144`
- it stayed frozen for `280` of `424` logged steps
- the policy kept asking for about `84-90 deg` reorientation
- the release condition never became true
- the robot hesitated, did not stab, and did not grasp

Interpretation:

- hard freeze prevented the dangerous burst
- but it also blocked the policy from completing the intended residue grasp


## Debugging Timeline

### 1. Add episode-level JSONL logging

Detailed episode logs were added under:

- `diffusion_policy_3d/env/juicing/episodes/`

Each step records:

- raw action
- unwrapped action `rpy`
- commanded target
- clipped command
- measured TCP before / after
- safety fault info

This made it possible to compare:

- policy output jumps
- commanded target jumps
- measured TCP jumps

### 2. Add command limiting and measured safety fallback

Safety was moved into the cartesian execution path so all command sources use
the same protection:

- `diffusion_policy_3d/env/juicing/xarm_wrapper.py`

Main behaviors:

- unwrap / compare orientations by true relative rotation
- limit position and orientation step sizes
- keep measured TCP position jump as the final hard fault

### 3. Split safety by stage

The task was then separated by execution profile:

- Stage 1: no safety fault
- Stage 2 fixed trajectory: bypass command protection
- Stage 3: keep safety and orientation control

This was necessary because:

- Stage 1 had no historical instability
- Stage 2 is scripted and should not be slowed or rejected
- Stage 3 contains the real instability source

### 4. Replace hard freeze with guarded rotation

The final Stage 3 behavior is:

- detect large wrist reorientation burst
- enter a short frozen window
- then transition into slow guarded rotation instead of staying fully locked

This keeps the "do not explode" property while still allowing the policy to
complete the residue grasp.


## Final Runtime Behavior

### Stage 1

- execution profile: `stage1_policy`
- no safety fault is applied
- this preserves the previously stable orange placement behavior

### Stage 2

- execution profile: `fixed_trajectory`
- command protection is bypassed by default
- this preserves the scripted pressing motion

### Stage 3

- execution profile: `stage3_policy`
- command step limiting stays enabled
- orientation gate watches for large wrist reorientation bursts

Current Stage 3 gate behavior:

1. If the requested wrist reorientation is large enough, switch into
   `frozen`.
2. Hold the current orientation for a short number of steps.
3. If the policy still insists on a large reorientation, do not hard-freeze
   forever. Instead, move toward the requested orientation with a smaller
   guarded rotation step.
4. Keep measured position jump as the final hard safety stop.

This is implemented in:

- `diffusion_policy_3d/env/juicing/juicing_env.py`

Key areas in that file:

- Stage 3 gate state definition
- `_apply_stage3_orientation_gate`
- `step()` replacing the raw command with `cmd_target_gated_mm_deg`


## Current Default Parameters

Current Stage 3 defaults in `juicing_all.yaml`:

- `stage3_max_rot_step_deg: 12.0`
- `stage3_orientation_gate_enabled: True`
- `stage3_orientation_burst_deg: 30.0`
- `stage3_orientation_burst_window_deg: 20.0`
- `stage3_orientation_burst_window_steps: 3`
- `stage3_orientation_burst_hits: 2`
- `stage3_orientation_freeze_steps: 6`
- `stage3_guarded_rot_step_deg: 4.0`
- `stage3_orientation_release_deg: 12.0`
- `stage3_orientation_release_stable_steps: 3`

Current scripts:

- full pipeline: `scripts/eval_juicing_dual_policy.sh`
- Stage 3 only: `scripts/eval_juicing_stage3_policy.sh`


## Validation Result

Latest operator feedback with the current guarded-rotation solution:

- about `20` trials
- about `1` failure
- no more "fly away" / explosion behavior
- Stage 3 wrist turning and grasping are a bit slower than before
- the slower motion is acceptable in practice
- the remaining observed failure came from not stabbing the residue deeply
  enough, so the residue could not be lifted

Interpretation:

- the critical safety issue is largely resolved
- the remaining weakness is task completion quality, not instability


## Rejected / Inadequate Approaches

These ideas were tested and should not be restored blindly:

- Fault directly on Euler-angle jump magnitude.
  This causes false positives because Euler branches can switch while the real
  orientation stays close.
- Use hard Stage 3 freeze as the final behavior.
  This avoids the burst but can leave the robot hesitating forever.
- Apply the same protection to all stages.
  This hurts Stage 1 and Stage 2 without solving the real problem source.


## Recommended Next Step

Do not redesign the mechanism unless instability returns.

If a small speed improvement is needed, prefer a conservative adjustment of only
one parameter:

- increase `stage3_guarded_rot_step_deg` slightly, for example from `4.0` to
  `5.0`

This is lower risk than changing the trigger thresholds or removing the Stage 3
gate.
