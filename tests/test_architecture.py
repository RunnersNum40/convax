import ast
import inspect
from pathlib import Path

import pytest

from convax import (
    AbstractConvexSet,
    AbstractPointContainmentSet,
    AbstractSupportSet,
    AffineImage,
    AxisAlignedBounds,
    ConstrainedZonotope,
    ConvexHull,
    Ellipsoid,
    HalfspacePolyhedron,
    MinkowskiSum,
    SupportResult,
    VertexPolytope,
    Zonotope,
)

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "convax"
SOURCE_FILES = tuple(SOURCE_ROOT.rglob("*.py"))
FINAL_CLASSES = (
    AffineImage,
    AxisAlignedBounds,
    ConstrainedZonotope,
    ConvexHull,
    Ellipsoid,
    HalfspacePolyhedron,
    MinkowskiSum,
    SupportResult,
    VertexPolytope,
    Zonotope,
)


def is_jit_decorator(decorator: ast.expr) -> bool:
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    return (isinstance(decorator, ast.Name) and decorator.id == "jit") or (
        isinstance(decorator, ast.Attribute) and decorator.attr in {"jit", "filter_jit"}
    )


@pytest.mark.parametrize(
    "abstract_class",
    [AbstractConvexSet, AbstractPointContainmentSet, AbstractSupportSet],
)
def test_abstract_classes_cannot_be_instantiated(abstract_class: type) -> None:
    with pytest.raises(TypeError):
        abstract_class()


@pytest.mark.parametrize("concrete_class", FINAL_CLASSES)
def test_concrete_classes_are_final(concrete_class: type) -> None:
    assert getattr(concrete_class, "__final__", False)
    assert not inspect.isabstract(concrete_class)


@pytest.mark.parametrize("source_file", SOURCE_FILES, ids=lambda path: path.name)
def test_source_avoids_forbidden_python_constructs(source_file: Path) -> None:
    syntax_tree = ast.parse(source_file.read_text(), filename=str(source_file))
    forbidden_loops = tuple(
        node
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.For | ast.AsyncFor | ast.While | ast.comprehension)
    )
    super_calls = tuple(
        node
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "super"
    )
    jit_decorators = tuple(
        decorator
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        for decorator in node.decorator_list
        if is_jit_decorator(decorator)
    )
    assert not forbidden_loops
    assert not super_calls
    assert not jit_decorators
