Create one horizontal animation strip for Codex pet `tv-engineer`, state `jumping`.

Use the attached canonical base for identity. Use the attached layout guide only for slot count, spacing, centering, and padding; do not draw the guide.

Output exactly 5 full-body frames in one left-to-right row on flat pure magenta #FF00FF. Treat the row as 5 invisible equal-width slots: one centered complete pose per slot, evenly spaced, with no overlap, clipping, empty slots, labels, or borders.

Identity: same pet in every frame: Flat vector cartoon Codex pet, chibi engineer body, vintage rounded television head, the TV screen displays emoji expressions instead of facial features, default expression is a relaxed smile, compact full body silhouette, clean geometric limbs, neutral tech palette, simple shoes and gloves, no text, no background, no scenery. Dragging states must look like the pet is being lifted by the mouse with a startled emoji. The working state must show a six-frame story: start debugging, pick up an orange safety helmet, put it on, then sit on the ground facing a small computer and actively debug with a focused emoji. The failed state must already wear the orange safety helmet and look frustrated or sweaty. Only running and failed may show the orange safety helmet. Other states must not show the helmet. Keep the character identity locked across all states.. Preserve silhouette, face, proportions, markings, palette, material, style, and props.
Style: Pet-safe sprite: compact full-body mascot, readable in a 192x208 cell, clear silhouette, simple face, stable palette/materials, and crisp edges for chroma-key extraction. Style `flat-vector`: Flat vector-style mascot with simple geometric forms, crisp color areas, clean outline, and minimal shading. User style notes: Cute but readable at small size, crisp outline, minimal shading, expressive emoji screen, no extra props except a small computer in the running state..
Animation continuity: keep apparent pet scale and baseline stable within the row unless the state itself intentionally changes vertical position, such as `jumping`. Move the pose within the slot instead of redrawing the pet larger or smaller frame to frame.

State action: Hover jump loop: anticipation, lift, airborne peak, descent, and settle through body height.

State requirements:
- Show the jump through pose and vertical body position only: anticipation, lift, airborne peak, descent, settle.
- Do not draw ground shadows, contact shadows, drop shadows, oval shadows, landing marks, dust, smears, bounce pads, or motion marks under the pet.
- Keep the background outside the pet perfectly flat chroma key with no darker key-colored patches.

Clean extraction: crisp opaque edges, safe padding, no scenery, text, guide marks, checkerboard, shadows, glows, motion blur, speed lines, dust, detached effects, stray pixels, or chroma-key colors inside the pet.
