# escher-tile-deformer-examples

Minimal C++ demos for Escher-style tile deformation using a closed-form vector field.

This is a demo code for Emerging Technologies, Technical Paper - "Escher Tile Deformation via Closed-Form Solution" presented in SIGGRAPH 2025. [Event Link](https://s2025.conference-schedule.org/presentation/?id=misc_210&sess=sess592)

---

## Build Instructions

```bash
mkdir build && cd build
cmake ..
make
```

This builds two executables: `anteater` and `lamp`.

---

## Prerequisites

* CMake ≥ 3.12
* C++17 compiler
* OpenCV 4.x
* SDL2
* OpenMP (only for `lamp`)

> On macOS (Homebrew): `opencv`, `sdl2`, and `libomp` should be installed.

---

## How to Run

### 1) `anteater`

```bash
./anteater ../assets/anteater1.png ../assets/anteater2.png ../assets_background/bus.png
```

### 2) `lamp`

```bash
./lamp ../assets/Cartoon_25_lamp_purple_adjusted.png ../assets/Cartoon_25_lamp_yellow_adjusted.png ../assets/Cartoon_25_lamp_purple_adjusted.png ../assets/Cartoon_25_lamp_yellow_adjusted.png ../assets_background/empty.png
```

---

## Controls

* **Deform**: click-drag inside the mini preview (bottom-left)
* **Zoom**: mouse wheel
* **Save**: press **`S`** → writes PNGs to `exports/` with a timestamp
  (deformed tiles and unit image)


##  How to run the sticker scripts

```bash
pip install pillow

# From repo root
cd get_stickets

# For Anteater
python sticker_creator_anteater_tile_big.py --in 20250811_034125

# For Lamp
python sticker_creator_lamp_tile_big.py --in 20250811_030553
```

### Where the results go

* Images are written to: `build/exports/`

  * `sticker_sheet_anteater_<timestamp>.png`
  * `sticker_sheet_lamp_<timestamp>.png`

*(Both scripts expect source PNGs in `build/exports/` and a header at `header/header.jpeg`.)*


## Important files and folders

* `anteater.cpp` — example 1
* `lamp.cpp` — example 2
* `CMakeLists.txt` — builds both targets
* `build/exports/` — created automatically on first save
* `get_stickers/` — contains scripts for creating the stickers


## Bibliography
```
@inproceedings{Chen_2025, series={SIGGRAPH Conference Papers ’25},
  title={Escher Tile Deformation via Closed-Form Solution},
  url={http://dx.doi.org/10.1145/3721238.3730681},
  DOI={10.1145/3721238.3730681},
  booktitle={Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
  publisher={ACM},
  author={Chen, Crane He and Kim, Vladimir},
  year={2025},
  month=jul, pages={1–11},
  collection={SIGGRAPH Conference Papers ’25}
}
```
