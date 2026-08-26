import ast
import inspect
import tomllib
from pathlib import Path

import pytest

import convax
from convax import operations, sets
from convax.sets import (
    AbstractAffineMapClosedSet,
    AbstractAffinePreimageClosedSet,
    AbstractConvexHullClosedSet,
    AbstractConvexSet,
    AbstractIntersectionClosedSet,
    AbstractMinkowskiSumClosedSet,
    AbstractNegationClosedSet,
    AbstractPointContainmentSet,
    AbstractSupportSet,
    AbstractTranslationClosedSet,
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

PROJECT_ROOT = Path(__file__).parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "convax"
SOURCE_FILES = tuple(SOURCE_ROOT.rglob("*.py"))
ALGEBRA_SOURCE = SOURCE_ROOT / "operations" / "_algebra.py"
TRANSFORMATIONS_SOURCE = SOURCE_ROOT / "operations" / "_transformations.py"
OPERATION_SOURCES = (ALGEBRA_SOURCE, TRANSFORMATIONS_SOURCE)
PUBLIC_API_SOURCE_FILES = (
    *OPERATION_SOURCES,
    *tuple((SOURCE_ROOT / "sets").glob("_*.py")),
)
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
FINAL_SET_CLASSES = (
    AffineImage,
    ConstrainedZonotope,
    ConvexHull,
    Ellipsoid,
    HalfspacePolyhedron,
    MinkowskiSum,
    VertexPolytope,
    Zonotope,
)
SET_PRODUCING_OPERATIONS = {
    "affine_map",
    "affine_preimage",
    "convex_hull",
    "intersection",
    "minkowski_sum",
    "negate",
    "project_coordinates",
    "translate",
}
CLOSED_OPERATION_CAPABILITIES = {
    "affine_map": AbstractAffineMapClosedSet,
    "affine_preimage": AbstractAffinePreimageClosedSet,
    "convex_hull": AbstractConvexHullClosedSet,
    "intersection": AbstractIntersectionClosedSet,
    "minkowski_sum": AbstractMinkowskiSumClosedSet,
    "negate": AbstractNegationClosedSet,
    "translate": AbstractTranslationClosedSet,
}


def is_jit_decorator(decorator: ast.expr) -> bool:
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    return (isinstance(decorator, ast.Name) and decorator.id == "jit") or (
        isinstance(decorator, ast.Attribute) and decorator.attr in {"jit", "filter_jit"}
    )


def has_decorator(function: ast.FunctionDef, name: str) -> bool:
    for decorator in function.decorator_list:
        if isinstance(decorator, ast.Call):
            decorator = decorator.func
        if isinstance(decorator, ast.Name) and decorator.id == name:
            return True
    return False


def user_parameter_names(function: ast.FunctionDef) -> tuple[str, ...]:
    positional_parameters = (*function.args.posonlyargs, *function.args.args)
    if positional_parameters and positional_parameters[0].arg in {"self", "cls"}:
        positional_parameters = positional_parameters[1:]
    return tuple(
        parameter.arg
        for parameter in (*positional_parameters, *function.args.kwonlyargs)
    )


def missing_args_documentation(
    docstring: str | None, parameter_names: tuple[str, ...]
) -> tuple[str, ...]:
    if not parameter_names:
        return ()
    if docstring is None or "Args:" not in docstring.splitlines():
        return parameter_names
    documented_parameters = {
        line.strip().split(":", maxsplit=1)[0]
        for line in docstring.splitlines()
        if line.startswith("    ") and ":" in line
    }
    return tuple(name for name in parameter_names if name not in documented_parameters)


@pytest.mark.parametrize(
    "abstract_class",
    [
        AbstractAffineMapClosedSet,
        AbstractAffinePreimageClosedSet,
        AbstractConvexSet,
        AbstractConvexHullClosedSet,
        AbstractIntersectionClosedSet,
        AbstractMinkowskiSumClosedSet,
        AbstractNegationClosedSet,
        AbstractPointContainmentSet,
        AbstractSupportSet,
        AbstractTranslationClosedSet,
    ],
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


@pytest.mark.parametrize(
    "operation_source", OPERATION_SOURCES, ids=lambda path: path.name
)
def test_operation_dispatch_does_not_depend_on_concrete_representations(
    operation_source: Path,
) -> None:
    syntax_tree = ast.parse(
        operation_source.read_text(), filename=str(operation_source)
    )
    imported_set_names = {
        alias.name
        for node in syntax_tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "convax.sets"
        for alias in node.names
    }
    concrete_representations = {
        "ConstrainedZonotope",
        "Ellipsoid",
        "HalfspacePolyhedron",
        "VertexPolytope",
        "Zonotope",
    }
    assert imported_set_names.isdisjoint(concrete_representations)


def test_set_producing_methods_match_closed_operation_capabilities() -> None:
    for concrete_class in FINAL_SET_CLASSES:
        for operation_name, capability in CLOSED_OPERATION_CAPABILITIES.items():
            assert hasattr(concrete_class, operation_name) is issubclass(
                concrete_class, capability
            )
            assert not hasattr(concrete_class, f"_{operation_name}")
            if hasattr(concrete_class, operation_name):
                assert not getattr(
                    getattr(concrete_class, operation_name), "__override__", False
                )
        assert not hasattr(concrete_class, "project_coordinates")

    assert set(convax.__all__) == {"operations", "sets"}
    assert set(operations.__all__) >= SET_PRODUCING_OPERATIONS
    assert set(sets.__all__) >= {
        concrete_class.__name__ for concrete_class in FINAL_CLASSES
    }


def test_runtime_typechecking_is_ci_only() -> None:
    configuration = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    project_dependencies = configuration["project"]["dependencies"]
    development_dependencies = configuration["dependency-groups"]["dev"]
    pytest_arguments = configuration["tool"]["pytest"]["addopts"]

    assert not any(
        dependency.startswith("typeguard") for dependency in project_dependencies
    )
    assert any(
        dependency.startswith("typeguard") for dependency in development_dependencies
    )
    assert "--jaxtyping-packages=convax,typeguard.typechecked" in pytest_arguments


@pytest.mark.parametrize(
    "source_file", PUBLIC_API_SOURCE_FILES, ids=lambda path: path.name
)
def test_public_callables_document_arguments(source_file: Path) -> None:
    syntax_tree = ast.parse(source_file.read_text(), filename=str(source_file))
    missing_documentation: list[str] = []

    for definition in syntax_tree.body:
        if isinstance(definition, ast.FunctionDef):
            if definition.name.startswith("_") or has_decorator(definition, "overload"):
                continue
            missing_parameters = missing_args_documentation(
                ast.get_docstring(definition), user_parameter_names(definition)
            )
            if missing_parameters:
                missing_documentation.append(
                    f"{definition.name}: {', '.join(missing_parameters)}"
                )
            continue

        if not isinstance(definition, ast.ClassDef) or definition.name.startswith("_"):
            continue

        for method in definition.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            if method.name == "__init__":
                docstring = ast.get_docstring(definition)
            elif method.name.startswith("_"):
                continue
            else:
                docstring = ast.get_docstring(method)
            missing_parameters = missing_args_documentation(
                docstring, user_parameter_names(method)
            )
            if missing_parameters:
                missing_documentation.append(
                    f"{definition.name}.{method.name}: {', '.join(missing_parameters)}"
                )

    assert not missing_documentation
