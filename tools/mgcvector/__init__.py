"""Isolated managed execution for experimental broadcast-scalar vector shaders."""

from .runtime import VECTOR_TARGET, VectorExecutionError, VectorShaderExecutor

__all__ = ["VECTOR_TARGET", "VectorExecutionError", "VectorShaderExecutor"]
