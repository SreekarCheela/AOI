# Hardware Bill of Materials — ESP32 Vision Rig

Design/procurement deliverable for [Issue #2](https://github.com/SreekarCheela/AOI/issues/2), part of the
[shadow-mode epic](https://github.com/SreekarCheela/AOI/issues/1). This is a **paper BOM**: specific,
purchasable part numbers with vendors, approximate prices, and datasheet links, produced so the
BOM owner / line owner can review and confirm before anything is ordered. Nothing here has been
physically procured or bench-tested.

Prices are street prices at time of writing (August 2026) in USD, single-unit qty. Re-check before
ordering — several categories here (Jetson dev kits in particular) have moved significantly in 2026.

> **Camera resolution is a placeholder pending confirmation.** Every choice in section 1 assumes the
> epic's placeholder finest-feature size of ~0.3–0.5mm (0402/0201 passives) from pre-flight checklist
> item 4. See [DESIGN.md](./DESIGN.md#assumptions-pending-confirmation) for what changes if the real
> BOM-owner answer differs.

---

## 1. ESP32 camera module (capture-only, per epic architecture)

| Item | Part # | Vendor | Price (approx) | Datasheet |
|---|---|---|---|---|
| MCU dev board | ESP32-S3-DevKitC-1-N16R8 (ESP32-S3-WROOM-1-N16R8, 16MB flash / 8MB PSRAM) | [Digi-Key](https://www.digikey.com/en/products/detail/espressif-systems/ESP32-S3-DEVKITC-1-N16R8/16162037) / [Mouser](https://www.mouser.com/ProductDetail/Espressif-Systems/ESP32-S3-DevKitC-1-N16R8) | ~$14 | [ESP32-S3-DevKitC-1 datasheet](https://docs.espressif.com/projects/esp-dev-kits/en/latest/esp32s3/esp32-s3-devkitc-1/user_guide.html) |
| Camera sensor module | Arducam OV5640 5MP autofocus camera module, DVP parallel interface, M12 lens mount (Arducam part B0431, "ESP32 OV5640" variant) | [Arducam.com](https://www.arducam.com/) / [Amazon](https://www.amazon.com/) | ~$30–35 | [OV5640 sensor datasheet](https://cdn.sparkfun.com/datasheets/Sensors/LightImaging/OV5640_datasheet.pdf) |
| Macro/close-focus M12 lens | Arducam M12 lens kit, 8mm/12mm interchangeable, macro-focus option | [Arducam.com](https://www.arducam.com/product-category/m12-lens/) | ~$15–20 | [Arducam M12 lens guide](https://docs.arducam.com/Nvidia-Jetson-Camera/Native-Camera/M12-Lens-Kit/) |
| PSRAM/flash | Included on N16R8 variant above (needed for JPEG framebuffer at 5MP) | — | included | — |

**Subtotal: ~$60–70/unit**

Why ESP32-S3 (not the older ESP32/ESP32-CAM): the S3's LCD_CAM peripheral and 8MB PSRAM comfortably
buffer a 5MP JPEG frame; the classic ESP32-CAM (OV2640, 2MP, 4MB PSRAM) does not have the resolution
or memory headroom for the target feature size. See [DESIGN.md](./DESIGN.md) for the resolution math.

**Bench-only alternative** (cheaper, lower resolution, useful for early wiring/trigger-chain bench
tests before the real camera arrives): Freenove ESP32-WROVER CAM board w/ OV2640, ~$20 on Amazon.
Not sufficient for final resolution validation — placeholder only.

---

## 2. Trigger sensor (board-presence detection on conveyor)

| Item | Part # | Vendor | Price (approx) | Datasheet |
|---|---|---|---|---|
| **Recommended — production-grade** | Omron E3F2-DS30C4-M diffuse photoelectric sensor, M18 threaded barrel, 300mm sensing range, 12–24VDC, NPN, dark-ON/light-ON selectable | [Digi-Key](https://www.digikey.com/en/products/detail/omron-automation-and-safety/E3F2-DS30C4-M/1776922) / [Radwell](https://www.radwell.com/Buy/OMRON/OMRON/E3F2-DS30C4-M) | ~$100–120 | [Omron E3F2 series datasheet](https://www.ia.omron.com/products/family/558/) |
| Bench-test / bring-up alternative | Generic IR break-beam or diffuse IR proximity module (Sharp GP2Y0A21 or similar), 3.3–5V logic-level output | [SparkFun](https://www.sparkfun.com/) / [Adafruit](https://www.adafruit.com/) | ~$5–10 | [Sharp GP2Y0A21YK0F datasheet](https://cdn.sparkfun.com/datasheets/Sensors/Proximity/gp2y0a21yk_e.pdf) |

**Subtotal: ~$100–120/unit (production), ~$10 (bench)**

Rationale: the epic explicitly assumes no existing "board present" PLC signal, so this is a new,
independent sensor. A diffuse photoelectric (IR) sensor is preferred over a mechanical limit switch
because it's non-contact — no wear, no risk of catching/displacing boards on the conveyor, and it
tolerates board-to-board size variation without needing a precise mechanical trip point. NPN output
is chosen to match typical 24V industrial I/O conventions; the ESP32 side needs an optocoupler or
level-shifting stage (see [DESIGN.md](./DESIGN.md) wiring) since the ESP32 GPIO is 3.3V-only and not
5V/24V tolerant.

The cheap IR module is for **bench bring-up only** — validating trigger→capture timing logic before
committing to the industrial sensor and before line-side mounting (which needs the checklist item 5
safety sign-off regardless of sensor choice).

---

## 3. Custom LED lighting rig

| Item | Part # | Vendor | Price (approx) | Datasheet |
|---|---|---|---|---|
| Dark-field ring (primary) | Adafruit NeoPixel Ring — 24 x 5050 RGBW LED w/ Integrated Drivers, Natural White (~4500K), Product ID 2853 | [Adafruit](https://www.adafruit.com/product/2853) | ~$25 | [Adafruit NeoPixel Ring guide](https://learn.adafruit.com/adafruit-neopixel-uberguide) |
| Diffuse fill (secondary) | Second Adafruit NeoPixel Ring (same part, mounted at a higher on-axis angle) | [Adafruit](https://www.adafruit.com/product/2853) | ~$25 | (same as above) |
| Diffuser material | Frosted acrylic or PTFE diffuser sheet, cut to ring ID/OD | [McMaster-Carr](https://www.mcmaster.com/) | ~$10 | — |
| Driver / level shift | 74AHCT125 logic-level shifter (3.3V ESP32 GPIO → 5V NeoPixel data) | [Adafruit](https://www.adafruit.com/product/1787) | ~$2 | [74AHCT125 datasheet](https://www.ti.com/lit/ds/symlink/sn74ahct125.pdf) |

**Subtotal: ~$60/unit**

**Production upgrade path (if DIY rings prove insufficient during bench validation):** dedicated
machine-vision white LED ring light, e.g. CCS Inc. LDR2-70WH2 (white LED ring, high-CRI, uniform
current drive, no PWM flicker at camera shutter speeds), ~$180–250 from
[CCS America](https://ccsamerica.com/) or [Advanced Illumination](https://www.advancedillumination.com/).
Flagged as a likely upgrade, not the starting BOM — see rationale below.

Rationale: solder joints are small, curved, highly specular (shiny) surfaces when good, and duller/
grainier when cold or disturbed — this reflectivity difference is the primary visual signal the
model needs, more so than color. Two rings at different angles give **dark-field + diffuse** lighting:
- **Low-angle ring** (mounted near-grazing to the board, ~15–30° from the board plane): good solder
  fillets act like small curved mirrors and throw a bright specular highlight back at low angles;
  cold/dull joints scatter light diffusely and appear dark by comparison. This is the primary defect
  signal — it's the same principle industrial solder AOI systems use (multi-angle/dome lighting to
  separate specular from diffuse reflection).
- **Higher-angle/near-axis ring**: even fill lighting to avoid deep shadows around taller components,
  so the low-angle ring's contrast isn't lost in an otherwise-black frame.

NeoPixel rings are chosen for the *bench validation* phase specifically because they're
individually-addressable and PWM-brightness-controllable from the ESP32 with a couple of GPIO pins —
useful for iterating on angle/brightness/color-temperature empirically against real sample joints
before committing to a fixed industrial light (acceptance criterion #2 in the issue: "bench-validated
against sample solder joints"). Natural white (~4500K) is chosen over RGB color-mixing to keep color
rendering consistent and avoid hue artifacts on solder (which is not the visual feature being read).

---

## 4. Edge compute — Jetson Orin Nano dev kit

| Item | Part # | Vendor | Price (approx) | Datasheet |
|---|---|---|---|---|
| Dev kit | NVIDIA Jetson Orin Nano Super Developer Kit (945-13766-0000-000) | [NVIDIA](https://developer.nvidia.com/embedded/buy-jetson) / [Seeed Studio](https://www.seeedstudio.com/) / [Amazon](https://www.amazon.com/) | **~$399** (see note) | [Jetson Orin Nano Super datasheet](https://developer.nvidia.com/embedded/jetson-orin-nano-devkit-user-guide) |
| Storage | NVMe M.2 SSD, 256GB+ (e.g. WD Black SN770) — dev kit needs boot media, no onboard eMMC on this SKU | [Digi-Key](https://www.digikey.com/) / [Amazon](https://www.amazon.com/) | ~$30 | — |

**Subtotal: ~$430/unit**

⚠️ **Price flag for BOM owner / budget approval:** the Orin Nano Super Dev Kit launched at $249 in
December 2024. NVIDIA raised prices across the Jetson lineup by up to ~100% in mid-2026 — this kit is
now listing around $399 at most retailers as of July 2026 ([CNX Software coverage](https://www.cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101/),
[JetsonHacks original announcement](https://jetsonhacks.com/2024/12/17/jetson-orin-nano-super-developer-kit/)).
A couple of retailers (e.g. SparkFun) still list old-price backordered stock that's unlikely to
fulfill at that price — budget for ~$400, not $249.

Rationale: this matches the epic's explicit architecture decision (on-prem inference, no cloud
round-trip, for IP/confidentiality reasons) and its "Jetson Orin Nano-class" spec. The Super Dev Kit
gives 67 TOPS INT8 / up to 1024 CUDA cores after the 2024 firmware unlock — comfortably enough
headroom for a whole-board classification CNN (the MVP model scope per the epic), with room to grow
into per-joint localization if that fast-follow is later greenlit.

---

## 5. Mounting hardware (bracket/enclosure)

| Item | Part # | Vendor | Price (approx) | Datasheet |
|---|---|---|---|---|
| T-slot extrusion frame | 8020 Inc. 1010 series aluminum T-slot extrusion, cut to length | [8020.net](https://8020.net/) | ~$8/ft | [8020 1010 series spec](https://8020.net/1010.html) |
| Corner brackets | 8020 Inc. 4-hole inside corner bracket, 10 series | [8020.net](https://8020.net/) | ~$5 each (need ~6–8) | — |
| Camera/sensor arm mount | Adjustable articulating mount, 1/4-20 thread (standard camera mount) | [SmallRig](https://www.smallrig.com/) / Amazon | ~$15–25 | — |
| Electronics enclosure | Hammond 1554D2GY, ABS, IP54-rated, ~120x120x60mm | [Digi-Key](https://www.digikey.com/en/products/detail/hammond-manufacturing/1554D2GY/1815776) | ~$18 | [Hammond 1554 series datasheet](https://www.hammfg.com/files/parts/pdf/1554.pdf) |
| Cable glands | PG7/PG9 nylon cable glands, IP68, 5-pack | [Digi-Key](https://www.digikey.com/) / Amazon | ~$8 | — |

**Subtotal: ~$100–130**

Rationale: 80/20 T-slot is chosen because the exact mounting geometry (camera standoff distance,
lighting angle) can't be finalized until bench validation against real sample joints — T-slot lets the
whole rig be re-positioned without redesigning brackets. IP54 enclosure for the ESP32 + power/signal
wiring protects against the dust/particulate environment typical near a conveyor; the Jetson itself
should stay outside a sealed enclosure (or in a ventilated one) since it needs airflow for its fan.
**Final bracket geometry is explicitly out of scope for this BOM pass** — it depends on physical
clearance at the install site, which is checklist item 5 (line safety/guarding sign-off) territory,
not something that can be nailed down from a desk.

---

## 6. Power supply & network isolation

| Item | Part # | Vendor | Price (approx) | Datasheet |
|---|---|---|---|---|
| Jetson power supply | Mean Well GST60A19-P1J, 19V/3.16A (60W), isolated, DIN-rail/desktop | [Digi-Key](https://www.digikey.com/en/products/detail/mean-well-usa-inc/GST60A19-P1J/7702123) | ~$25 | [Mean Well GST60A series datasheet](https://www.meanwell.com/Upload/PDF/GST60A/GST60A-SPEC.PDF) |
| ESP32 / sensor / lighting power supply | Mean Well LRS-50-24, 24V/2.2A (50W), DIN-rail | [Digi-Key](https://www.digikey.com/en/products/detail/mean-well-usa-inc/LRS-50-24/7703458) | ~$20 | [Mean Well LRS-50 datasheet](https://www.meanwell.com/Upload/PDF/LRS-50/LRS-50-SPEC.PDF) |
| Buck converter (24V→5V for ESP32/NeoPixel) | Pololu 5V, 5A step-down regulator D24V50F5 | [Pololu](https://www.pololu.com/product/2851) | ~$10 | [D24V50F5 datasheet](https://www.pololu.com/product/2851) |
| Network isolation switch | StarTech IES51000, 5-port unmanaged industrial gigabit switch, DIN-rail/wall mount, IP30 | [StarTech](https://www.startech.com/en-us/networking-io/ies51000) / [Amazon](https://www.amazon.com/) | ~$180–220 | [StarTech IES51000 datasheet](https://www.startech.com/en-us/networking-io/ies51000) |

**Subtotal: ~$235–275**

Rationale: the epic explicitly calls for an independent power supply (not tapped from line control
power) and network isolation from the plant OT/PLC network — both are direct architecture
requirements, not just good practice here. A dedicated 5-port switch gives the ESP32 and Jetson their
own physical LAN segment; nothing about the epic requires managed/VLAN features for shadow mode (no
PLC integration, no live gating), so an unmanaged industrial switch is sufficient and cheaper than a
managed one. **This is explicitly provisional pending checklist item 6** — if plant IT says the image
traffic must route through existing infrastructure (rather than a fully air-gapped segment), a
managed switch with VLAN tagging would be needed instead; budget difference is roughly +$100–150.

---

## Total estimated cost (single bench unit)

| Section | Subtotal |
|---|---|
| 1. ESP32 + camera | ~$65 |
| 2. Trigger sensor (production) | ~$110 |
| 3. Lighting rig | ~$60 |
| 4. Jetson Orin Nano Super Dev Kit | ~$430 |
| 5. Mounting hardware | ~$115 |
| 6. Power + network isolation | ~$255 |
| **Total** | **~$1,035** |

This is a single bench/prototype unit for the capture-chain validation described in the issue's
acceptance criteria — not a per-line-station production cost (no redundancy, spares, or
install-labor line items included).
