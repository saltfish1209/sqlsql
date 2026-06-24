Create one horizontal animation strip for Codex pet `tv-engineer`, state `waiting`.

Use the attached canonical base for identity. Use the attached layout guide only for slot count, spacing, centering, and padding; do not draw the guide.

Output exactly 6 full-body frames in one left-to-right row on flat pure magenta #FF00FF. Treat the row as 6 invisible equal-width slots: one centered complete pose per slot, evenly spaced, with no overlap, clipping, empty slots, labels, or borders.

Identity: same pet in every frame: Flat vector cartoon Codex pet, chibi engineer body, vintage rounded television head, the TV screen displays emoji expressions instead of facial features, default expression is a relaxed smile, compact full body silhouette, clean geometric limbs, neutral tech palette, simple shoes and gloves, no text, no background, no scenery. Dragging states must look like the pet is being lifted by the mouse with a startled emoji. The working state must show a six-frame story: start debugging, pick up an orange safety helmet, put it on, then sit on the ground facing a small computer and actively debug with a focused emoji. The failed state must already wear the orange safety helmet and look frustrated or sweaty. Only running and failed may show the orange safety helmet. Other states must not show the helmet. Keep the character identity locked across all states.. Preserve silhouette, face, proportions, markings, palette, material, style, and props.
Style: Pet-safe sprite: compact full-body mascot, readable in a 192x208 cell, clear silhouette, simple face, stable palette/materials, and crisp edges for chroma-key extraction. Style `flat-vector`: Flat vector-style mascot with simple geometric forms, crisp color areas, clean outline, and minimal shading. User style notes: Cute but readable at small size, crisp outline, minimal shading, expressive emoji screen, no extra props except a small computer in the running state..
Animation continuity: keep apparent pet scale and baseline stable within the row unless the state itself intentionally changes vertical position, such as `jumping`. Move the pose within the slot instead of redrawing the pet larger or smaller frame to frame.

State action: Needs-input loop: expectant asking pose for approval, help, or user input.

State requirements:
- Show that Codex needs approval, help, or user input through an expectant asking pose.
- Keep the motion patient and readable, without turning it into ordinary idle or review.

Clean extraction: crisp opaque edges, safe padding, no scenery, text, guide marks, checkerboard, shadows, glows, motion blur, speed lines, dust, detached effects, stray pixels, or chroma-key colors inside the pet.
