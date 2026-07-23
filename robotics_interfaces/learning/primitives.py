"""JSON-safe primitive types and conversion for learning contracts.

Only plain Python values are representable: None, bool, int, float, str,
nested lists/tuples of those, and mappings with str keys.  ``to_primitive``
converts dataclasses, enums, tuples and mappings into that shape and fails
explicitly for anything else -- objects are never silently coerced with
``str()``.

No robotics_sim, Qt, numpy, torch or pandas imports are allowed here.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Mapping, Union

# A JSON-safe primitive value.  Tuples are normalised to lists on
# conversion so the result can be fed directly to ``json.dumps``.
Primitive = Union[None, bool, int, float, str, list, dict]


class UnsupportedPrimitiveError(TypeError):
    """Raised when a value cannot be represented as a JSON-safe primitive."""


def to_primitive(value: Any) -> Primitive:
    """Convert ``value`` into a JSON-safe primitive structure.

    Supported inputs: ``None``, ``bool``, ``int``, ``float``, ``str``,
    dataclass instances, ``enum.Enum`` members, ``tuple``/``list``
    sequences, and mappings with ``str`` keys.  Anything else raises
    :class:`UnsupportedPrimitiveError` instead of being silently converted
    to a string.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        # Enum payloads must themselves be primitive; the name is included
        # so two enums with identical values still hash differently.
        return {"enum": type(value).__name__, "name": value.name, "value": to_primitive(value.value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_primitive(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (tuple, list)):
        return [to_primitive(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Primitive] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise UnsupportedPrimitiveError(
                    f"Mapping keys must be str, got {type(key).__name__}: {key!r}"
                )
            result[key] = to_primitive(item)
        return result
    raise UnsupportedPrimitiveError(
        f"Cannot convert object of type {type(value).__name__} to a JSON-safe primitive"
    )
