"""
Navigation debug diagnostics — a neutral, immutable contract describing the
values planner/navigation/safety/controller code actually used to make a
decision, for consumption by a GUI overlay.

This package must never import Qt, robotics_sim.app, or
robotics_sim.simulation.engine. It is pure Python: dataclasses, an Enum, and
a small bounded ring buffer. Producers (robotics_sim.planning,
robotics_sim.environment, robotics_sim.control, robotics_sim.simulation)
fill these structures with values they already computed; consumers (the GUI
canvas) only ever read them.
"""
