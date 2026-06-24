Create one horizontal animation strip for Codex pet `tvbuddy`, state `running-right`.

Use the attached canonical base for identity. Use the attached layout guide only for slot count, spacing, centering, and padding; do not draw the guide.

Output exactly 8 full-body frames in one left-to-right row on flat pure magenta #FF00FF. Treat the row as 8 invisible equal-width slots: one centered complete pose per slot, evenly spaced, with no overlap, clipping, empty slots, labels, or borders.

Identity: same pet in every frame: flat vector cartoon Codex pet, chibi engineer body, retro rounded television head, the TV screen shows emoji expressions instead of facial features, clean silhouette, neutral tech palette, friendly and readable at small size, default relaxed smile emoji, no text, no background, no hard hat except when explicitly requested by state. Preserve silhouette, face, proportions, markings, palette, material, style, and props.
Style: Pet-safe sprite: compact full-body mascot, readable in a 192x208 cell, clear silhouette, simple face, stable palette/materials, and crisp edges for chroma-key extraction. Style `flat-vector`: Flat vector-style mascot with simple geometric forms, crisp color areas, clean outline, and minimal shading. User style notes: simple geometric shapes, crisp outline, minimal shading, cute engineer proportions.
Animation continuity: keep apparent pet scale and baseline stable within the row unless the state itself intentionally changes vertical position, such as `jumping`. Move the pose within the slot instead of redrawing the pet larger or smaller frame to frame.

State action: Mouse-dragged-right loop: the pet is being lifted and dragged to the right by an unseen mouse, with a shocked emoji on the TV screen and a dangling picked-up body pose.

State requirements:
- Show directional drag movement to the right through body, limb, and prop movement only.
- The pet must look like it is being picked up by the mouse: feet off the ground, body dangling slightly upward, arms reacting to being lifted.
- The TV screen must show a shocked or frightened emoji in every frame.
- Do not add a visible mouse cursor, hand, string, hook, or any external grabbing device.
- Do not show a hard hat in this state.
- The row must unmistakably face and travel right.
- The movement cadence must alternate visibly across the 8 frames instead of repeating one nearly static stride.
- Do not draw speed lines, dust clouds, floor shadows, motion trails, or detached motion effects.

Clean extraction: crisp opaque edges, safe padding, no scenery, text, guide marks, checkerboard, shadows, glows, motion blur, speed lines, dust, detached effects, stray pixels, or chroma-key colors inside the pet.
