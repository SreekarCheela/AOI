# Example output

Produced by `python cli.py demo`, no hardware involved.

- **banding_example.png** — the core idea. Three lighting angles coloured
  red/green/blue. A correctly formed solder joint sweeps blue → green → red;
  a joint starved of solder never develops the steep face and loses the red.
- **defect_map.png** — a full inspected board. 12 faults were injected;
  all 12 were found with no good part wrongly flagged. Red = missing,
  orange = misplaced, yellow = crooked, blue = short between pads,
  pink = standing on end.
- **channels.png** — the four raw captures every test works from.
