# edge-edge-visualizer

Interactive visualizer for edge-edge motion and signed separation between two box edges, with a live Gauss-map inset.

**[Live web build](https://randygaul.github.io/edge-edge-visualizer/)**

## Controls

| Input          | Action                                     |
| -------------- | ------------------------------------------ |
| Slider drag    | Scrub time `t` in `[0, 1]`                 |
| Left-drag      | Orbit the main camera                      |
| Right-drag     | Virtual-trackball rotate the nearest box   |
| Scroll wheel   | Zoom (main view, or Gauss-map inset)       |
| `R`            | Reset scene                                |
| `Esc`          | Quit                                       |

## Build (Windows)

Requires Visual Studio + CMake. Runs `cmake` with the default generator.

```
build.bat       REM configure + build
vis.bat         REM build + launch
```

Output: `build\Release\gauss_map_viz.exe`.

## Build (Web)

Requires [emsdk](https://github.com/emscripten-core/emsdk) cloned as a sibling directory (`../emsdk`).

```
web.cmd
```

Output: `build_web\index.html` (+ `index.js`, `index.wasm`). Serve over HTTP:

```
cd build_web && python -m http.server
```

Then open `http://localhost:8000/`.

## Deploying the web build to GitHub Pages

Copy `build_web\index.{html,js,wasm}` to a `gh-pages` branch (or `docs/` on `main`) and enable Pages for the repo in GitHub settings. The demo link above assumes Pages is served at the repo root.
