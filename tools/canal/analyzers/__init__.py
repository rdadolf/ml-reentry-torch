"""Analyzer registry — maps analysis names to Analysis instances."""

from tools.canal.analyzers.base import Analysis
from tools.canal.analyzers.breaks import BreaksAnalysis
from tools.canal.analyzers.codegen import CodegenAnalysis
from tools.canal.analyzers.fusion import FusionAnalysis
from tools.canal.analyzers.fx import FXAnalysis
from tools.canal.analyzers.ir import IRAnalysis
from tools.canal.analyzers.passes import PassesAnalysis

ANALYZERS: dict[str, Analysis] = {
    "fx": FXAnalysis(),
    "ir": IRAnalysis(),
    "codegen": CodegenAnalysis(),
    "breaks": BreaksAnalysis(),
    "fusion": FusionAnalysis(),
    "passes": PassesAnalysis(),
}
