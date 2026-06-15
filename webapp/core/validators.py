#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Input Validators Module

Validates user inputs and converts between different formats.
"""

from typing import Tuple


class CollapseTargetValidator:
    """Validate and convert collapse target values."""

    @staticmethod
    def parse_collapse_target(value: float, num_nodes: int) -> Tuple[float, str]:
        """
        Parse and validate collapse target input.

        Args:
            value: User input value
            num_nodes: Total number of nodes in network

        Returns:
            - Normalized ratio (0-1)
            - Description string

        Raises:
            ValueError: If input is invalid
        """
        if value <= 0:
            raise ValueError("The value must be greater than 0.")

        if 0 < value < 1:
            # Ratio mode
            ratio = value
            absolute = int(ratio * num_nodes)
            description = f"{ratio:.2%} (about {absolute} nodes)"

        elif 1 <= value <= num_nodes:
            # Absolute mode
            absolute = int(value)
            ratio = absolute / num_nodes
            description = f"{absolute} nodes ({ratio:.2%})"

        else:
            raise ValueError(
                f"Invalid collapse target.\n\n"
                f"Input value: {value}\n"
                f"Network size: {num_nodes} nodes\n\n"
                f"Use one of these formats:\n"
                f"  - Ratio mode: 0 < x < 1, for example 0.3 means 30%.\n"
                f"  - Absolute mode: 1 <= x <= {num_nodes}, for example 150 means 150 nodes."
            )

        return ratio, description

    @staticmethod
    def parse_warning_target(value: int, num_nodes: int) -> Tuple[float, str]:
        """
        Parse and validate warning target input.

        Args:
            value: User input (integer, 1 to num_nodes)
            num_nodes: Total number of nodes

        Returns:
            - Normalized ratio (0-1)
            - Description string

        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(value, (int, float)) or value != int(value):
            raise ValueError("Warning Target must be an integer.")

        value = int(value)

        if not (1 <= value <= num_nodes):
            raise ValueError(
                f"Warning Target is out of range.\n\n"
                f"Input value: {value}\n"
                f"Valid range: 1 to {num_nodes}\n\n"
                f"The decision threshold is Warning Target divided by the initial network size."
            )

        ratio = value / num_nodes
        description = f"{value} step(s) / {num_nodes} nodes = {ratio:.4f}"

        return ratio, description


class AttackSequenceValidator:
    """Validate attack sequence data."""

    @staticmethod
    def validate_sequence(sequence: list, G_nodes: set) -> Tuple[list, dict]:
        """
        Validate attack sequence against network nodes.

        Args:
            sequence: List of node IDs to attack
            G_nodes: Set of valid node IDs in the network

        Returns:
            - Valid sequence (nodes that exist in network)
            - Statistics dictionary
        """
        valid_sequence = []
        invalid_nodes = []

        for node in sequence:
            if node in G_nodes:
                valid_sequence.append(node)
            else:
                invalid_nodes.append(node)

        stats = {
            'total': len(sequence),
            'valid': len(valid_sequence),
            'invalid': len(invalid_nodes),
            'invalid_nodes': invalid_nodes[:10],  # First 10 invalid nodes
        }

        return valid_sequence, stats
