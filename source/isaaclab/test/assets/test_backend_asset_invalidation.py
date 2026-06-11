# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression checks for backend asset writer invalidation helpers."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


BACKENDS = ("isaaclab_newton", "isaaclab_physx", "isaaclab_ovphysx")

DATA_CLASSES = {
    "articulation/articulation_data.py": "ArticulationData",
    "rigid_object/rigid_object_data.py": "RigidObjectData",
    "rigid_object_collection/rigid_object_collection_data.py": "RigidObjectCollectionData",
}

WRITER_EXPECTATIONS = {
    "articulation/articulation.py": {
        "write_root_link_pose_to_sim_index": ("reset_pose",),
        "write_root_link_pose_to_sim_mask": ("reset_pose",),
        "write_root_com_pose_to_sim_index": ("reset_pose",),
        "write_root_com_pose_to_sim_mask": ("reset_pose",),
        "write_root_com_velocity_to_sim_index": ("reset_velocity",),
        "write_root_com_velocity_to_sim_mask": ("reset_velocity",),
        "write_root_link_velocity_to_sim_index": ("reset_velocity",),
        "write_root_link_velocity_to_sim_mask": ("reset_velocity",),
        "write_joint_position_to_sim_index": ("reset_pose", "reset_velocity"),
        "write_joint_position_to_sim_mask": ("reset_pose", "reset_velocity"),
        "write_joint_velocity_to_sim_index": ("reset_velocity",),
        "write_joint_velocity_to_sim_mask": ("reset_velocity",),
    },
    "rigid_object/rigid_object.py": {
        "write_root_link_pose_to_sim_index": ("reset_pose",),
        "write_root_link_pose_to_sim_mask": ("reset_pose",),
        "write_root_com_pose_to_sim_index": ("reset_pose",),
        "write_root_com_pose_to_sim_mask": ("reset_pose",),
        "write_root_com_velocity_to_sim_index": ("reset_velocity",),
        "write_root_com_velocity_to_sim_mask": ("reset_velocity",),
        "write_root_link_velocity_to_sim_index": ("reset_velocity",),
        "write_root_link_velocity_to_sim_mask": ("reset_velocity",),
    },
    "rigid_object_collection/rigid_object_collection.py": {
        "write_body_link_pose_to_sim_index": ("reset_pose",),
        "write_body_link_pose_to_sim_mask": ("reset_pose",),
        "write_body_com_pose_to_sim_index": ("reset_pose",),
        "write_body_com_pose_to_sim_mask": ("reset_pose",),
        "write_body_com_velocity_to_sim_index": ("reset_velocity",),
        "write_body_com_velocity_to_sim_mask": ("reset_velocity",),
        "write_body_link_velocity_to_sim_index": ("reset_velocity",),
        "write_body_link_velocity_to_sim_mask": ("reset_velocity",),
    },
}


def _parse_backend_file(backend: str, relative_path: str) -> ast.Module:
    return ast.parse((REPO_ROOT / "source" / backend / backend / "assets" / relative_path).read_text())


def _find_class(module: ast.Module, name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Could not find class {name}.")


def _find_method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Could not find method {name}.")


def _called_methods(function_node: ast.FunctionDef) -> set[str]:
    calls = set()
    for node in ast.walk(function_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)
    return calls


def _reachable_called_methods(class_node: ast.ClassDef, method_name: str) -> set[str]:
    method_nodes = {node.name: node for node in class_node.body if isinstance(node, ast.FunctionDef)}
    reachable_calls = set()
    pending_methods = [method_name]
    visited_methods = set()

    while pending_methods:
        current_method = pending_methods.pop()
        if current_method in visited_methods:
            continue
        visited_methods.add(current_method)
        direct_calls = _called_methods(method_nodes[current_method])
        reachable_calls.update(direct_calls)
        pending_methods.extend(call for call in direct_calls if call in method_nodes and call not in visited_methods)

    return reachable_calls


def test_backend_asset_data_classes_expose_reset_helpers():
    """Asset data classes expose shared pose and velocity invalidation helpers."""
    for backend in BACKENDS:
        for relative_path, class_name in DATA_CLASSES.items():
            class_node = _find_class(_parse_backend_file(backend, relative_path), class_name)
            method_names = {node.name for node in class_node.body if isinstance(node, ast.FunctionDef)}
            assert "reset_pose" in method_names, f"{backend}/{relative_path} is missing reset_pose()."
            assert "reset_velocity" in method_names, f"{backend}/{relative_path} is missing reset_velocity()."


def test_backend_asset_writers_delegate_invalidation_to_data_helpers():
    """State writers delegate stale-cache handling to the data class helpers."""
    for backend in BACKENDS:
        for relative_path, expectations in WRITER_EXPECTATIONS.items():
            module = _parse_backend_file(backend, relative_path)
            class_node = next(node for node in module.body if isinstance(node, ast.ClassDef))
            for method_name, expected_calls in expectations.items():
                _find_method(class_node, method_name)
                calls = _reachable_called_methods(class_node, method_name)
                for expected_call in expected_calls:
                    assert expected_call in calls, (
                        f"{backend}/{relative_path}:{method_name} does not call {expected_call}()."
                    )
