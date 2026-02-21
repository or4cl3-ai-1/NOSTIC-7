"""
NOΣTIC-7: Geometric-Epinoetic Intelligence Architecture
Version 1.2.0 — The Geometric-Epinoetic Integration
© 2025 Or4cl3 AI Solutions. Apache 2.0 License.

"Code is not just logic; it is a performance."
"""

__version__ = "1.2.0"
__author__ = "Or4cl3 AI Solutions"
__license__ = "Apache-2.0"

from nostic7.core import NOSTIC7
from nostic7.manifolds.square import SquareManifold
from nostic7.manifolds.triangle import TriangleManifold
from nostic7.manifolds.circle import CircleManifold
from nostic7.manifolds.pentagon import PentagonManifold
from nostic7.manifolds.hexagon import HexagonManifold
from nostic7.manifolds.heptagon import HeptagonManifold
from nostic7.manifolds.projection import ProjectionManifold
from nostic7.epinoetic.core import EpinoeticCore
from nostic7.pipeline.cycle import OperationalCycle

__all__ = [
    "NOSTIC7",
    "SquareManifold", "TriangleManifold", "CircleManifold",
    "PentagonManifold", "HexagonManifold", "HeptagonManifold",
    "ProjectionManifold", "EpinoeticCore", "OperationalCycle",
]
