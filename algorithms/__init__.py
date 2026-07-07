"""External algorithm plugins for multi-robot-frontier-lab.

Algorithm packages placed here should expose a `plugin.py` module with a
`create_plugin()` function returning an object compatible with
`robotics_interfaces.plugins.CoordinationPlugin`.

Algorithms must not import robotics_sim, Qt, MainWindow, canvas, or engine.
"""
