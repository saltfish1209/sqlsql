Create one horizontal animation strip for Codex pet `tv-head-engineer`, state `running-left`.

Use the attached canonical base for identity. Use the attached layout guide only for slot count, spacing, centering, and padding; do not draw the guide.

Output exactly 8 full-body frames in one left-to-right row on flat pure magenta #FF00FF. Treat the row as 8 invisible equal-width slots: one centered complete pose per slot, evenly spaced, with no overlap, clipping, empty slots, labels, or borders.

Identity: same pet in every frame: Television-head engineer mascot, full body, small compact proportions, flat vector cartoon style, clean crisp silhouette, simple limbs, engineer vibe. Default screen face is a relaxed smiling emoji. Screen shows emotion only with emoji, never text. No background and no extra scene elements. No computer except in running state. Orange safety helmet only appears in running and failed states; the helmet-wearing process appears only in running. Running-right and running-left are drag poses only, showing the mascot being lifted and pulled by the mouse, with a shocked emoji on the screen. Waving is friendly. Jumping is small and lively. Failed is frustrated or sweaty with orange safety helmet. Waiting is calm with slight lean or folded arms. Review is focused and leaning forward. Keep the same character identity, proportions, palette, and prop rules across all states.. Preserve silhouette, face, proportions, markings, palette, material, style, and props.
Style: Pet-safe sprite: compact full-body mascot, readable in a 192x208 cell, clear silhouette, simple face, stable palette/materials, and crisp edges for chroma-key extraction. Style `flat-vector`: Flat vector-style mascot with simple geometric forms, crisp color areas, clean outline, and minimal shading. User style notes: Flat vector cartoon, clean silhouette, minimal shading, crisp outline, no background scene, optimized for tiny animation readability..
Animation continuity: keep apparent pet scale and baseline stable within the row unless the state itself intentionally changes vertical position, such as `jumping`. Move the pose within the slot instead of redrawing the pet larger or smaller frame to frame.

State action: Dragging-left loop: show directional movement to the left through body and limb poses only.

State requirements:
- Show directional drag movement to the left through body, limb, and prop movement only.
- The row must unmistakably face and travel left.
- The movement cadence must alternate visibly across the 8 frames instead of repeating one nearly static stride.
- Do not draw speed lines, dust clouds, floor shadows, motion trails, or detached motion effects.
- This is not normal running. Show the mascot being lifted and dragged to the left by the mouse as a "picked up" pose, with the body slightly dangling rather than walking.
- The screen face must switch to a shocked emoji in every frame of this row.
- Do not add the orange safety helmet or a computer in this row.

Clean extraction: crisp opaque edges, safe padding, no scenery, text, guide marks, checkerboard, shadows, glows, motion blur, speed lines, dust, detached effects, stray pixels, or chroma-key colors inside the pet.
