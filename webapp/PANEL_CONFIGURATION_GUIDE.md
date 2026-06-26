# Monitoring Panel Configuration

Panel code lives in `webapp/visualization/panels.py`.

## Current Panels

- `create_lcc_panel()`: LCC Size against attack step, with the collapse target line.
- `create_collapse_distance_panel()`: Collapse Distance against attack step, with the decision threshold and warning marker.
- `create_natural_connectivity_panel()`: natural connectivity against attack step.
- `create_robustness_panel()`: R(DCR) against attack step.

## Style

The panels use a light Plotly theme to match the Streamlit app theme:

- template: `plotly_white`
- paper/plot background: white
- shared grid and axis styling through `_base_layout()` and `_axis_style()`
- responsive x-axis tick spacing through `_step_dtick()`

R(DCR) uses an automatic y-axis range so the full curve is visible. LCC Size remains fixed at `0` to `1.05` because it is a normalized proportion.

## Warning Display

The real-time warning rule is fixed to Collapse Distance:

```text
Collapse Distance <= Warning Target / N0
```

`create_collapse_distance_panel()` draws the decision threshold and marks the first step where the warning condition is reached.

## Editing Tips

For small visual changes, update the color values in each trace. For global layout changes, update `_base_layout()` and `_axis_style()` first so all panels stay consistent.
