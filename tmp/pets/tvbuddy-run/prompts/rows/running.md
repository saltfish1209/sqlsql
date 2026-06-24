Create one horizontal animation strip for Codex pet `tvbuddy`, state `running`.

Use the attached canonical base for identity. Use the attached layout guide only for slot count, spacing, centering, and padding; do not draw the guide.

Output exactly 6 full-body frames in one left-to-right row on flat pure magenta #FF00FF. Treat the row as 6 invisible equal-width slots: one centered complete pose per slot, evenly spaced, with no overlap, clipping, empty slots, labels, or borders.

Identity: same pet in every frame: flat vector cartoon Codex pet, chibi engineer body, retro rounded television head, the TV screen shows emoji expressions instead of facial features, clean silhouette, neutral tech palette, friendly and readable at small size, default relaxed smile emoji, no text, no background, no hard hat except when explicitly requested by state. Preserve silhouette, face, proportions, markings, palette, material, style, and props.
Style: Pet-safe sprite: compact full-body mascot, readable in a 192x208 cell, clear silhouette, simple face, stable palette/materials, and crisp edges for chroma-key extraction. Style `flat-vector`: Flat vector-style mascot with simple geometric forms, crisp color areas, clean outline, and minimal shading. User style notes: simple geometric shapes, crisp outline, minimal shading, cute engineer proportions.
Animation continuity: keep apparent pet scale and baseline stable within the row unless the state itself intentionally changes vertical position, such as `jumping`. Move the pose within the slot instead of redrawing the pet larger or smaller frame to frame.

State action: GPT-working loop: a 6-frame mini story showing the pet entering debugging mode, putting on an orange safety helmet, then sitting on the ground and debugging on a small computer with a focused emoji.

State requirements:
- Frame sequence requirement: start with no helmet, then pick up an orange safety helmet, then put it on, then end with the pet wearing the helmet while sitting on the ground debugging on a small computer.
- The computer may appear only in this state as a small simple prop directly in front of the pet.
- The final working poses must clearly show the pet sitting on the ground, facing the computer, actively debugging.
- The TV screen must shift from neutral/focused into a concentrated working emoji by the later frames.
- Show the pet actively working or processing, as if running a task: focused posture, busy hands or paws, purposeful bobbing, thinking motion, tool or prop motion only if already part of the pet identity, or other non-locomotion activity.
- Do not show literal foot-running, jogging, sprinting, treadmill motion, raised knees, long steps, pumping arms, directional travel, speed lines, dust clouds, floor shadows, motion trails, or detached motion effects.

Clean extraction: crisp opaque edges, safe padding, no scenery, text, guide marks, checkerboard, shadows, glows, motion blur, speed lines, dust, detached effects, stray pixels, or chroma-key colors inside the pet.
