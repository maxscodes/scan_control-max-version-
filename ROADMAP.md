# scan_control Roadmap

## Goal

Turn scan_control into an all-in-one STM control and automation program. It should run sequences of scans, spectroscopy and lithography without anyone intervening. It should keep a complete history of what the tip has done. In time it should be able to take a general goal and plan and run the experiments to reach it.

We are building it in small, testable chunks. Each feature runs with a human watching before it is trusted to run alone. Where a working Python script already exists, we port its tested Nanonis call order and safety logic rather than rewriting from scratch.

## Pillars

These are in rough dependency order.

| # | Pillar | What it means |
|---|--------|---------------|
| 1 | **Spectroscopy** | Bias spectroscopy (I(V), dI/dV) and Z spectroscopy, each usable as a sequence step |
| 2 | **Plotting and saving** | Every experiment is graphed in the UI and saved to disk automatically |
| 3 | **Unattended sequences** | Mixed sequences of scans, spectroscopy and lithography. Human approval at first, later fully autonomous |
| 4 | **Tip history** | A log of everything the tip does, with images pasted into a global map |
| 5 | **ML image analysis** | Judge image quality, detect tip damage and trigger tip-shaping to repair it |
| 6 | **H-desorption lithography** | Load a GDS pattern, plan a tip path and desorb hydrogen from the chosen areas |
| 7 | **ML litho QA** | Judge lithography quality and decide whether to fix or redo it |
| 8 | **Goal-driven autonomy** | Accept high-level goals, then plan and run the experiments |

## Status

| Feature | Pillar | State | Hardware-tested |
|---------|--------|-------|-----------------|
| Scan frame control and scan window (start, pause, resume, stop) | — | Done | Yes |
| Image sequences | 3 | Done (`sequence/`) | Yes |
| Coarse motion with auto approach and withdraw | — | Done | Yes |
| GDS map layer with scale multiplier | 6 | Done | Yes |
| I(V) series at stepped Z, with plots and `.dat` export | 1, 2 | Done (`spectroscopy/`) | **No** |
| GDS lithography: 5 path planners, Ramp or Z-lift moves, STOP, map overlay | 6 | Done (`lithography/`) | **No** |
| Ring auto-centering: estimator and display-only Ring Assist overlay | 5 | Steps 1 and 2 done (`analysis/ring_center.rs`, `ring_assist.rs`) | Overlay is not yet live-checked |

Only one tool can drive the tip at a time: a sequence, spectroscopy, lithography or a running scan.

## Next chunks

These are suggested and in a rough order. Each one gets agreed on before work starts.

1. **Hardware-test the I(V) series and lithography.** See the checklists below.
2. **Add spectroscopy as a sequence step.** Make an I(V) series a step type in the sequence editor, and save the data and plots automatically for each step.
3. **Add lithography as a sequence step.** Do this once lithography is trusted on hardware.
4. **Ring auto-centering, step 3: tip movement.** This waits until the overlay gives the right direction on 10 to 20 real scans. It includes the fallback rule: when a scan can't give a direction, move half way back toward the last good scan.
5. **Z spectroscopy.** Port the Python script if one exists.
6. **Common data store.** Put experiment results (scans, spectra, litho runs) in one place, with metadata, so the tip history and ML can use them.
7. **Tip history log.** Record a timestamped list of every action, with position, bias and setpoint, linked to images on the map.
8. **Tip-shaping routines.** Port the coarse tip-drift and tip-shaping routine (TipLift, BiasPulse, auto approach) that was left out of the lithography port.
9. **Approval gates in sequences.** Let a sequence pause for the operator to confirm before risky steps. This is the stage before full autonomy.

After that come the ML pillars (5 and 7), then goal-driven autonomy (8).

## Deferred or left out on purpose

- **Coarse tip-shaping from the Python lithography GUI.** This belongs to the tip-repair pillar, item 8 above.
- **Saving the lithography plan as JSON.** Not needed yet.
- **SXM overlay from the Python lithography GUI.** The app already pastes scans onto the map, so it isn't needed.

## Hardware-test checklists

### I(V) series

- [ ] Run a single curve with feedback on between curves, and check that the channels are found.
- [ ] Run a short series (3 curves, small Z step) and check that Z steps and verifies correctly.
- [ ] Abort mid-series and check that feedback stays in a safe state.
- [ ] Save the curves and compare the `.dat` format with the Python tool's output.

### Lithography

- [ ] Preview a path on a small frame and check that it lines up with the GDS on the map.
- [ ] Use "Move tip to path start" and check that the tip goes where the overlay says.
- [ ] Write a tiny pattern with modest values, using Ramp transitions.
- [ ] Write the same pattern using Z-lift transitions, and check the lift and restore readbacks.
- [ ] Press STOP mid-run and check that bias, setpoint, feedback and Follow Me speed are all restored.
- [ ] Check a rotated frame: the path should follow the red frame drawn on the map.
- [ ] On the v5e build, check that `FolMe.XYPosSet` and `FolMe.SpeedGet` work.

### Ring Assist

- [ ] Watch the overlay on live scans and record how often its direction is right.

## Design principles

- **Port tested Python logic.** Keep its Nanonis call order and safety checks.
- **Fail closed.** On a timeout, mismatch or error, stop and restore a safe state rather than carrying on.
- **Restore what you change.** Bias, setpoint, feedback and speeds go back to their earlier values afterwards.
- **One tool drives the tip at a time.**
- **Build for what comes later.** Sequence step types, data storage, the tip-history log and ML decision hooks should be shaped so later pillars plug in without rewrites.
