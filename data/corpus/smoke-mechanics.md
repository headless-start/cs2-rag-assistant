# Smoke mechanics in CS2

CS2's move to the Source 2 engine changed smokes more than any other mechanic,
and understanding them is part of modern CS2 knowledge.

## Volumetric smokes
Smokes are now a single dynamic volume of physical particles rather than a flat
2D sprite. The cloud expands to fill the space it is in — flowing down stairs,
bulging through doorways, and respecting the actual geometry of the room — so
the same smoke can look and behave differently depending on where it lands.

## Reacting to gunfire and explosions
Bullets passing through a smoke briefly push the particles aside, opening a thin
sliver of visibility along the line of fire. More importantly, an HE grenade
detonating inside or next to a smoke blows a large, temporary hole in it,
clearing a window to push or shoot through for a moment before the cloud
re-forms. Teams use this deliberately: throw a smoke to block an angle, then HE
it to create a brief gap on your terms.

## Duration and lighting
A smoke lasts about 18 seconds before dissipating. Smokes also interact with
light, so a cloud can be lit from one side, which affects how well it hides a
player crossing through it.

## Practical impact
Because smokes fill volume, a single well-placed smoke can seal a doorway more
reliably than in the old engine, but it can also seep into places you did not
intend. Line-ups from the previous game often had to be relearned for CS2.
