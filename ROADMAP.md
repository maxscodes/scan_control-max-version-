# scan_control Roadmap

*The master document: everything built so far, what is missing, and the plan. Updated 2026-11-17 (v1.1).*

**Goal.** Turn scan_control into an all-in-one STM control and automation program. It should run sequences of scans, spectroscopy and lithography without anyone intervening, keep a complete history of what the tip has done, and in time take a general goal and plan and run the experiments to reach it.

**How we build it.** In small, testable chunks. Each feature runs with a human watching before it is trusted to run alone. Where a working Python script already exists, we port its tested Nanonis call order and safety logic rather than rewriting from scratch.

**Contents:** [Progress](#progress) · [Next steps](#next-steps) · [Pillars](#pillars) (done and missing for each) · [Foundation](#foundation) · [Hardware-test checklists](#hardware-test-checklists) · [Deferred](#deferred-or-left-out-on-purpose) · [Design principles](#design-principles) · [Keeping this document up to date](#keeping-this-document-up-to-date)

---

## Progress

```
Overall                   ███████░░░░░░░░░░░░░  36 %
```

**Features: 67 built · 12 tested on the instrument · 55 built but not yet tested · 28 still missing**

```
 1 Spectroscopy            ███████████░░░░░░░░░  55 %
 2 Plotting and saving     █████████░░░░░░░░░░░  45 %
 3 Unattended sequences    ████████░░░░░░░░░░░░  40 %
 4 Tip history             ███████░░░░░░░░░░░░░  35 %
 5 ML image analysis       ███░░░░░░░░░░░░░░░░░  15 %
 6 H-desorption litho      ████████████░░░░░░░░  60 %
 7 ML litho QA             ░░░░░░░░░░░░░░░░░░░░   0 %
 8 Goal-driven autonomy    ░░░░░░░░░░░░░░░░░░░░   0 %
 9 Ring navigation         █████████░░░░░░░░░░░  45 %
10 Grid surveys            ████████████░░░░░░░░  60 %

   Foundation (not in the average)
   Map, scanning, motion   █████████████████░░░  85 %
```

**Markers used below:** ✅ built and tested on the instrument · 🔧 built, not yet tested on the instrument · ⬜ missing

The percentages are a judgement, checked against the code on 2026-11-17: how much of what each pillar describes exists, with 🔧 counting for less than ✅. Overall is the plain average of the ten pillars.

**The biggest single gain is testing.** 55 features are built but untested. A few sessions on the instrument working through the [checklists](#hardware-test-checklists) would move most pillars up by 10 to 20 points without writing new code.

---

## Next steps

In rough order. Each one gets agreed on before work starts.

| # | Step | Pillars | Why now |
|---|------|---------|---------|
| 1 | **Hardware-test what is built**: I(V) series, lithography, Fine Motion, sequence tip moves, safe survey tracking, Ring Assist on live scans, Polish on a real survey | 1, 3, 6, 9, 10 | Most built features wait on this; it unblocks steps 3 and 4 |
| 2 | **Z spectroscopy**: port the Python script if one exists, and make it a sequence step | 1 | The last missing kind of spectroscopy |
| 3 | **Lithography in sequences, with a record of each run** | 2, 3, 6 | Once lithography is trusted on hardware; the record is needed for litho QA later |
| 4 | **Hands-off ring centring**: move without Proceed, act on the fallback, stop on the smallest ring, centring as a sequence step | 9 | Once the overlay is right on 10 to 20 real scans |
| 5 | **Common data store**: scans, spectra and litho runs in one place with metadata | 2, 4, 5, 7 | Tip history and the ML pillars build on it |
| 6 | **Tip history log**: a timestamped list of every tip action (moves, bias, setpoint, approach, litho, pulses) linked to images on the map | 4 | Uses the data store |
| 7 | **Tip Check chunks 2 to 4**: tune on labelled archive scans, a learned model (Python training on `tip_training/`, ONNX in Rust), a TipCheck sequence step that can pause on a poor tip | 5, 10 | Lets surveys and sequences notice a bad tip |
| 8 | **Tip-shaping routines**: port TipLift, bias pulse and controlled indentation (left out of the lithography port), with a re-check afterwards | 5 | Repairs what Tip Check finds |
| 9 | **Approval gates in sequences**: pause for the operator before risky steps | 3 | The stage before full autonomy |
| 10 | **Survey around a ring centre**: centre first (Ring Assist), then a grid around it | 9, 10 | Combines the two |

After that come ML litho QA (pillar 7), then goal-driven autonomy (pillar 8).

---

## Pillars

| # | Pillar | What it means |
|---|--------|---------------|
| 1 | **Spectroscopy** | Bias spectroscopy (I(V), dI/dV) and Z spectroscopy, each usable as a sequence step |
| 2 | **Plotting and saving** | Every experiment is graphed in the UI and saved to disk automatically |
| 3 | **Unattended sequences** | Mixed sequences of scans, spectroscopy and lithography. Human approval at first, later fully autonomous |
| 4 | **Tip history** | A log of everything the tip does, with images pasted into a global map |
| 5 | **ML image analysis** | Judge image quality, detect tip damage and trigger tip shaping to repair it |
| 6 | **H-desorption lithography** | Load a GDS pattern, plan a tip path and desorb hydrogen from the chosen areas |
| 7 | **ML litho QA** | Judge lithography quality and decide whether to fix or redo it |
| 8 | **Goal-driven autonomy** | Accept high-level goals, then plan and run the experiments |
| 9 | **Ring navigation** | Analyse ring scans (where the frame is on the rings, which way the centre is) and centre the tip on a ring system automatically |
| 10 | **Grid surveys** | Scan a grid of overlapping tiles reliably, keep it on plan with closed-loop tracking, and turn it into one clean picture |

Only one tool can drive the tip at a time: a sequence, spectroscopy, lithography, centring or a running scan.

### 1 Spectroscopy (55 %)

Code: `connection/nanonis/spectroscopy/`, ported from the "Multiple I(V) experiment" Python tool.

Done:
- 🔧 I(V) series at stepped tip heights: Z controller off (optionally feedback on, settle, off between curves), reference Z read, one bias sweep per height, Z stepped and verified
- 🔧 dI/dV and log dI/dV from the lock-in channels (picked automatically, lock-in Y first, or chosen)
- 🔧 Live plots in the I(V) window (I, log I, dI/dV, log dI/dV), Start and Abort
- 🔧 Run from a sequence: an I(V) series at each tip position of an entry
- 🔧 Safe abort: stops after the curve in progress, leaves Z feedback in a safe state

Missing:
- ⬜ Z spectroscopy
- ⬜ Any of the above tested on the instrument

### 2 Plotting and saving (45 %)

Done:
- ✅ Scans shown live and stamped onto the map; "Save" stores the .sxm in the Nanonis session folder
- 🔧 I(V) curves saved automatically to the session folder: one .dat per curve, plus one combined plot and one table per series
- 🔧 Stamps renamed after the .sxm file Nanonis saved the same scan to, found by comparing the data (`stamp_names.rs`, `analysis/sxm_match.rs`)
- 🔧 Project (map, stamps and their results) saved to the session folder next to the day's scans ("Save to day folder", Ctrl+Shift+S)
- 🔧 Polished survey pictures saved as PNG or put on the map as an image layer
- 🔧 Survey grid calibration saved; coarse calibration runs keep their scans and a before/after PNG per move

Missing:
- ⬜ A record of each lithography run (only the drawn shapes can be saved)
- ⬜ Z spectroscopy plots and files (pillar 1)
- ⬜ One common data store for all results, with metadata (Next steps, 5)

### 3 Unattended sequences (40 %)

Code: `connection/nanonis/sequence/`.

Done:
- ✅ Image sequences: entries with frame, resolution, speed, bias, current, scan direction; withdraw, coarse move, approach, wait, scan, stamp
- ✅ Run all, run from the selected entry, pause, pause after the scan, continue, abort, restart an entry
- 🔧 Plan drawn on the map, frames dragged on the map, "+ Where the tip is", capture the current frame, save and load sequences
- 🔧 Warm-up scan after each coarse move
- 🔧 Time-to-completion estimate, learning how long hardware steps take (`eta.rs`)
- 🔧 Survives the app closing: the running sequence is saved after every step and picked up again after "Rebuild & restart" or a crash, while Nanonis keeps scanning (`checkpoint.rs`)
- 🔧 Tip moves (Follow Me, placed by Ctrl+click on the map) and an I(V) series at each position; entries can skip the scan
- (Survey grids: see pillar 10)

Missing:
- ⬜ Lithography as a step
- ⬜ Z spectroscopy and ring centring as steps
- ⬜ Approval gates before risky steps (only grid tracking asks today)
- ⬜ Running fully unattended, with recovery from errors

### 4 Tip history (35 %)

Done:
- ✅ Every scan stamped onto one global map, where it was taken
- 🔧 Coarse moves kept with the project and drawn on the map ("Show coarse move history", with frame boxes)
- 🔧 Every coarse move and drag correction logged for the calibration
- 🔧 Ring Assist and Tip Check results kept with each stamp
- 🔧 Stamps linked to their .sxm files

Missing:
- ⬜ A timestamped log of every tip action (moves, bias, setpoint, approach, litho, pulses) linked to the images
- ⬜ A way to browse or search that history

### 5 ML image analysis (15 %)

Code: `analysis/tip_quality/`, `analysis/tip_training.rs`, `connection/nanonis/tip_check.rs`.

Done:
- 🔧 Tip Check chunk 1 (rule-based, display only): multi-tip ghosts from the autocorrelation (fine detail, larger features, edges; lattice peaks removed), instability, forward/backward disagreement, bluntness
- 🔧 Verdict shown live with copy-offset arrows, and saved on stamps ("Judge stamps", "Show on stamps")
- 🔧 Archive labelling (G/D/M/B/U), evaluation, threshold tuning, features CSV export
- (Ring analysis: see pillar 9)

Missing:
- ⬜ Tip Check scored on real labelled scans
- ⬜ A learned model (Tip Check chunks 2 to 4)
- ⬜ Judging overall image quality (noise, drift, feedback trouble), not just the tip
- ⬜ Tip shaping (bias pulse, controlled indentation) triggered by a poor verdict

### 6 H-desorption lithography (60 %)

Code: `connection/nanonis/lithography/`, ported from the Python lithography GUI v1.18.

Done:
- ✅ GDS map layer with scale multiplier; load a litho GDS and pick its layer
- 🔧 Shapes drawn directly in the lithography window (circles, polygons by points), edited on the map, saved and loaded
- 🔧 5 path planners over the GDS polygons or shapes inside the scan frame, path preview on the map
- 🔧 Ramp or Z-lift moves between segments, "Move tip to path start"
- 🔧 Confirmed start, STOP, and everything restored afterwards (bias, setpoint, feedback, Follow Me speed); optional scan of the frame after a complete run

Missing:
- ⬜ Writing tested on the instrument
- ⬜ Lithography as a sequence step
- ⬜ A record of each run (what was written, where, with which settings)

### 7 ML litho QA (0 %)

Missing:
- ⬜ Everything: judging a written pattern from the scan after it, and deciding to fix or redo it. Needs lithography tested on hardware and run records first.

### 8 Goal-driven autonomy (0 %)

Missing:
- ⬜ Everything: taking a high-level goal and planning the experiments. Needs pillars 3, 5 and 7 first.

### 9 Ring navigation (45 %)

Code: `analysis/ring_center.rs`, `gds_prior.rs`, `ring_match.rs`, `ring_live.rs`, `landing.rs`, `ring_stamp.rs`, `ring_training.rs`, `connection/nanonis/ring_assist.rs`.

Ring analysis, done:
- 🔧 Ring centre estimate from one Z frame: arc or arm end, direction to the centre
- 🔧 GDS prior: direction from where the frame sits on the GDS, band direction check, GDS offset estimate with Shift GDS, warning when the GDS moves; matched 12/12 labels
- 🔧 GDS-guided ring matching (matched filter on the expected bands), with forward/backward option
- 🔧 Landing: where a frame is on the GDS without trusting the GDS placement; way to the centre right on 29/40 labelled frames
- 🔧 Results saved with stamps; existing stamps analysed by selection, group or all ("Analyse stamps")
- 🔧 Teach panel: label stamps (good / corrected / stop / smallest ring); retraining on the labels
- 🔧 Polish images lines survey tiles up on the GDS rings

Automatic centring, done:
- 🔧 Semi-auto centring: after each complete frame, proposes the coarse move toward the ring centre within move limits, makes it when you click Proceed, then waits for a frame scanned after the move

Missing:
- ⬜ Live checks: how often the direction is right on 10 to 20 real scans
- ⬜ Hands-off centring: moving without Proceed, stopping by itself on the smallest ring
- ⬜ The fallback when a frame is unsure (move half way back toward the last good frame): only drawn, not acted on
- ⬜ Centring as a sequence step

### 10 Grid surveys (60 %)

Code: `connection/nanonis/sequence/` (`grid_track.rs`), `analysis/register.rs`, `analysis/mosaic.rs`, `analysis/polish*.rs`, `polish_window.rs`.

Done:
- 🔧 Survey planner: a grid of tiles with overlap, snake order, appended as sequence entries
- 🔧 Closed-loop tracking: each tile placed by matching the one before, the next move corrected so tiles land where planned, motor step size learned and saved as a calibration
- 🔧 Made safe after the 2026-11-16 survey went wrong: only near the predicted move, look-alikes refused, step changes limited, corrected moves stay near the plan, asks before changing a move, every automatic fix undoable, done tiles never moved. Replayed on that survey's data: the look-alike is now refused
- 🔧 Tiles collected into the survey's layer group
- 🔧 Align stamps: line up overlapping stamps by their image, optionally undoing drift stretch; undoable

Polish images (v1.1), done:
- 🔧 Per tile: background and filters, edge trim, brightness, contrast, tilt, move, crop, hide, on top
- 🔧 Tiles matched by height, tilt, bow and contrast across overlaps and small gaps; seams healed (all, or Ctrl+click one); two-band seam blending
- 🔧 Alignment: auto with look-alike refusal, a fast seam-by-seam check by eye, rings lined up with the GDS; can move the stamps on the map
- 🔧 Contrast evened out, hills flattened, shared contrast stretch with histogram, colours, scale bar
- 🔧 Stretch and Alignment on/off toggles; several tiles selected and moved together, shown live while dragging
- 🔧 Save PNG, or put on the map as an image layer

Missing:
- ⬜ Safe tracking tested on the instrument
- ⬜ Polish images tried on a real survey
- ⬜ Pausing a survey on a poor tip (needs Tip Check in sequences, pillar 5)
- ⬜ Surveys around a ring centre found by Ring Assist (Next steps, 10)

---

## Foundation

Everything the pillars stand on. Not counted in the overall average.

**Map and project**
- ✅ One map holding the live scan region, stamps, GDS layers and images; pan, zoom, fit scan (Home), grid, scale bar
- ✅ Projects: open, save, save as, recent projects
- 🔧 Layers with groups (click a group to select it all), show and hide, per-image opacity
- 🔧 Undo and redo for map changes
- 🔧 Import: .sxm scans, PNG and JPEG images (anchor, flip, lock), GDS files
- 🔧 Light and dark themes; preferences remembered

**Scanning**
- ✅ Scan Control panel: start, pause, resume, stop, save; bias and setpoint; live current; progress
- ✅ Live scan downloaded and drawn on the map; stamp it at any time
- ✅ Display processing: background fits (mean, line, plane on the GPU; parabolas on the CPU), line fixes, smoothing and sharpening, colour scales

**Coarse motion and calibration**
- ✅ Coarse moves, auto approach and auto withdraw
- 🔧 Calibrate X / Y, and automated calibration runs (scan, step, scan, and back) measured from the image shift, with the last run's scans reviewable by eye
- 🔧 Calibration learned from drags: dragging a stamp to where it really is records a correction; "Last test only" or "Last test + on-the-fly"; the last test never overwritten, earlier tests restorable
- 🔧 Pair view: two overlapping scans laid over each other to check a move by eye

**Fine motion**
- 🔧 Move the scan frame within the piezo range (arrows step one frame, Ctrl+drag or click on the map, Back), optionally stamping first and starting the next scan
- 🔧 Move the tip with Follow Me (Go, live Ctrl+drag, Halt), kept inside the piezo range

**App**
- 🔧 Rebuild & restart from inside the app; the project reopens and a running sequence continues
- ✅ Nanonis TCP connection on separate workers (status, lines, frames, commands)
- Version rule: +0.1 per day with changes (see `src/version.rs`)

---

## Hardware-test checklists

Ticking these moves features from 🔧 to ✅.

### I(V) series

- [ ] Run a single curve with feedback on between curves, and check that the channels are found.
- [ ] Run a short series (3 curves, small Z step) and check that Z steps and verifies correctly.
- [ ] Abort mid-series and check that feedback stays in a safe state.
- [ ] Save the curves and compare the `.dat` format with the Python tool's output.
- [ ] Check the dI/dV plot picks the right lock-in channel.

### Lithography

- [ ] Preview a path on a small frame and check that it lines up with the GDS on the map.
- [ ] Use "Move tip to path start" and check that the tip goes where the overlay says.
- [ ] Write a tiny pattern with modest values, using Ramp transitions.
- [ ] Write the same pattern using Z-lift transitions, and check the lift and restore readbacks.
- [ ] Press STOP mid-run and check that bias, setpoint, feedback and Follow Me speed are all restored.
- [ ] Check a rotated frame: the path should follow the red frame drawn on the map.
- [ ] On the v5e build, check that `FolMe.XYPosSet` and `FolMe.SpeedGet` work.

### Fine Motion

- [ ] Move the frame with the scan stopped: check that Nanonis shows the new frame centre and that the dashed outline matched it.
- [ ] Press "Stop scan, stamp & move frame" mid-scan: check the stamp lands where the scan was, and the next scan starts if asked.
- [ ] Check the arrows on a rotated frame: each step should move one frame (less the overlap) in the arrow's direction on the map.
- [ ] Send the tip with Go, then Ctrl+drag it with "Move at once": check it follows at a safe Follow Me speed, and that Halt stops it.
- [ ] Check that the target can't leave the piezo range.

### Sequence tip moves and spectroscopy

- [ ] One entry, no scan, two tip positions with moves only: check the tip goes to each and the sequence waits for it.
- [ ] Scan, then one position with a 2-curve I(V) series: check the curves show in the I(V) window and the batch folder appears in the Nanonis session folder.
- [ ] Abort during a series: check it stops after the curve in progress, Z feedback is left OFF, and the curves taken so far are saved.
- [ ] After the series, check the next scan runs with Z feedback ON.
- [ ] Close the app mid-sequence and reopen it: check the sequence carries on from where it was.

### Ring navigation

- [ ] Watch the overlay on 10 to 20 live scans and record how often its direction is right.
- [ ] Try semi-auto centring: check each proposed move is sensible and that it waits for a fresh frame after moving.
- [ ] Run "Analyse stamps" on a past sequence group and compare the arrows with what the scans show.

### Grid surveys

- [ ] Run a small survey (3×3) with tracking on and "ask before correcting" on: check each tile lands where planned and every question makes sense.
- [ ] Drag a tile to where it really is mid-survey: check the next moves follow.
- [ ] Save the grid calibration at the end and check the next survey starts closer to plan.

### Polish images

- [ ] Polish a real survey grid: Auto-align, then "Line up rings with GDS", and check the rings run smoothly across the seams.
- [ ] Shift+drag several tiles together and check the live image follows the mouse.
- [ ] Toggle Stretch and Alignment off and on and check the picture and the saved PNG match what is shown.

### Coarse calibration

- [ ] Run an automated calibration and review its moves by eye.
- [ ] Drag a few stamps to where they really are and check the learned calibration improves the next moves.

---

## Deferred or left out on purpose

- **Coarse tip-shaping from the Python lithography GUI.** Belongs with tip shaping (Next steps, 8).
- **Saving the lithography plan as JSON.** Not needed yet.
- **SXM overlay from the Python lithography GUI.** The app already pastes scans onto the map.

---

## Design principles

- **Port tested Python logic.** Keep its Nanonis call order and safety checks.
- **Fail closed.** On a timeout, mismatch or error, stop and restore a safe state rather than carrying on.
- **Restore what you change.** Bias, setpoint, feedback and speeds go back to their earlier values afterwards.
- **One tool drives the tip at a time.**
- **Ask before acting, until trusted.** New automatic moves propose first and act after the operator agrees; hands-off comes after live checks.
- **Every automatic fix can be undone.**
- **Build for what comes later.** Sequence step types, data storage, the tip-history log and ML decision hooks should be shaped so later pillars plug in without rewrites.

---

## Keeping this document up to date

When something is built, tested or planned:

1. Add it under its pillar (or Foundation) with the right marker: ⬜ → 🔧 when built, 🔧 → ✅ when tested on the instrument.
2. Recount the feature line under Progress (count the ✅, 🔧 and ⬜ lines).
3. Nudge that pillar's bar (20 characters, one per 5 %) and recompute the overall average of the ten pillars.
4. Update Next steps if the order changes, and the date at the top.
