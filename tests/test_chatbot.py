
import ast
from pathlib import Path

import pytest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

# Load only the chatbot helper functions, without starting Streamlit.
SOURCE = APP_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)

FUNCTION_NAMES = {
    "empty_trip_preferences",
    "extract_trip_request",
    "recalculate_budget",
    "merge_trip_preferences",
}

FUNCTIONS = [
    node for node in TREE.body
    if isinstance(node, ast.FunctionDef)
    and node.name in FUNCTION_NAMES
]

import re

namespace = {
    "re": re,
    "MARKETS": {
        "Amsterdam": {"currency": "€"},
        "New York City": {"currency": "$"},
    },
}

module = ast.Module(body=FUNCTIONS, type_ignores=[])
exec(compile(module, str(APP_PATH), "exec"), namespace)

extract_trip_request = namespace["extract_trip_request"]
merge_trip_preferences = namespace["merge_trip_preferences"]


def test_total_budget_nyc():
    request = extract_trip_request(
        "I have $600 for 4 nights in NYC",
        "New York City",
    )
    assert request["nights"] == 4
    assert request["budget_total"] == pytest.approx(600)
    assert request["budget_night"] == pytest.approx(150)


def test_total_budget_amsterdam():
    request = extract_trip_request(
        "I have €800 for 4 nights in Amsterdam",
        "Amsterdam",
    )
    assert request["budget_total"] == pytest.approx(800)
    assert request["budget_night"] == pytest.approx(200)


def test_nightly_budget():
    request = extract_trip_request(
        "My budget is $200 per night for 4 nights",
        "New York City",
    )
    assert request["budget_night"] == pytest.approx(200)


def test_ambiguous_budget():
    request = extract_trip_request(
        "I have $500",
        "New York City",
    )
    assert request["budget_night"] is None


def test_conversation_memory():
    first = extract_trip_request(
        "I have $600 for 4 nights in NYC",
        "New York City",
    )
    second = extract_trip_request(
        "We are 4 people",
        "New York City",
    )
    merged = merge_trip_preferences(first, second)

    assert merged["guests"] == 4
    assert merged["nights"] == 4
    assert merged["budget_night"] == pytest.approx(150)
