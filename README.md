# escher-deform-examples

Minimal C++ demos for Escher-style tile deformation using a closed-form vector field.

---

## TL;DR (Build)

```bash
mkdir build && cd build
cmake ..
make
```

This builds two executables: `anteater` (xx pattern) and `lamp` (4*2 pattern).

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

### 1) `anteater` — xx example

```bash
./anteater ../assets/anteater1.png ../assets/anteater2.png ../assets_background/bus.png
```

### 2) `lamp` — 4*2* example

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

* `anteater.cpp` — 2-tile demo
* `lamp.cpp` — 4-tile OpenMP demo
* `CMakeLists.txt` — builds both targets
* `exports/` — created automatically on first save
