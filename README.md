# escher-tile-deformer-examples

Minimal C++ demos for Escher-style tile deformation using a closed-form vector field.

This is a demo code for Emerging Technologies, Technical Paper - "Escher Tile Deformation via Closed-Form Solution" presented in SIGGRAPH 2025. [Event Link](https://s2025.conference-schedule.org/presentation/?id=misc_210&sess=sess592)

---

## TL;DR (Build)

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
./lamp ../assets/Cartoon_25_lamp_blue_adjusted.png ../assets/Cartoon_25_lamp_blue_adjusted.png ../assets/Cartoon_25_lamp_blue_adjusted.png ../assets/Cartoon_25_lamp_blue_adjusted.png ../assets_background/living_room.png
```

---

## Controls

* **Deform**: click-drag inside the mini preview (bottom-left)
* **Zoom**: mouse wheel
* **Save**: press **`S`** → writes PNGs to `exports/` with a timestamp
  (deformed tiles and unit image)

---

## Repo Layout

* `anteater.cpp` — example 1
* `lamp.cpp` — example 2
* `CMakeLists.txt` — builds both targets
* `exports/` — created automatically on first save


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