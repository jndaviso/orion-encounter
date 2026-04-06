# Orion–Moon Trajectory Viewer

This project provides an interactive visualization of spacecraft trajectory data (e.g., Orion) alongside the Moon’s motion, using ephemeris data and physical geometry to analyze velocity and line-of-sight occultation.

The app is built with Streamlit and Matplotlib and is designed to run locally or be deployed on Streamlit Community Cloud.

---

## Features

- 2D trajectory visualization (X–Y plane, EME2000/J2000)
- Moon trajectory and dynamically positioned lunar disk (to scale)
- Real-time spacecraft state marker with heading orientation
- Velocity decomposition:
  - Tangential velocity
  - Radial velocity
  - Total 3D velocity
- Occultation analysis:
  - Detects whether the spacecraft is occluded by the Moon (3D geometry)
  - Displays tangent rays from Earth to the Moon
  - Shades the occlusion region
- Interactive controls:
  - Slider to scrub through ephemeris states
  - Play mode (continuous animation)
  - Auto-update mode (sync to real time)
- Optional diagnostic panel with detailed geometry outputs

---

## Project Structure

```text
.
├── app.py
├── requirements.txt
└── data/
    ├── orion_oem.asc
    └── de421.bsp
