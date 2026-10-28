import pytest
from unittest.mock import Mock
from typing import Optional
from hecstac.ras.item import RASModelItem


@pytest.fixture
def mock_manager() -> Mock:
    """Fixture providing a mock instance of RASModelItem for _primary_plan."""
    manager = Mock()
    manager.pf = Mock()
    manager.pf.plan_current = "basic_plan"
    manager.plan_assets = []

    # To bypass cached_property object is not callable runtime exception,
    # we dynamically bind the true function logic directly to the instance.
    manager._execute_primary_plan = RASModelItem._primary_plan.func
    return manager


def create_mock_asset(name: str, is_encroached: bool) -> Mock:
    """Helper to quickly generate mock PlanAsset items."""
    asset = Mock()
    asset.name = name
    asset.file = Mock()
    asset.file.is_encroached = is_encroached
    return asset


def test_no_plan_assets(mock_manager) -> None:
    """Returns None if plan_assets list is empty."""
    mock_manager.plan_assets = []
    result = mock_manager._execute_primary_plan(mock_manager)
    assert result is None


def test_single_plan_asset(mock_manager) -> None:
    """Returns the only asset if there is only 1 plan asset."""
    asset = create_mock_asset("Plan_100yr", is_encroached=True)
    mock_manager.plan_assets = [asset]
    result = mock_manager._execute_primary_plan(mock_manager)
    assert result == asset


def test_multiple_assets_one_matching_current(mock_manager) -> None:
    """Multiple unencroached plans exist; returns the asset matching pf.plan_current."""
    asset_100 = create_mock_asset("Plan_100yr", is_encroached=False)
    asset_500 = create_mock_asset("Plan_500yr", is_encroached=False)
    mock_manager.plan_assets = [asset_100, asset_500]
    mock_manager.pf.plan_current = "Plan_500yr"
    result = mock_manager._execute_primary_plan(mock_manager)
    assert result == asset_500


def test_multiple_assets_none_matching_current(mock_manager) -> None:
    """Multiple unencroached plans exist, but none match current plan. Returns first unencroached plan asset."""
    asset_100 = create_mock_asset("Plan_100yr", is_encroached=False)
    asset_500 = create_mock_asset("Plan_500yr", is_encroached=False)
    mock_manager.plan_assets = [asset_100, asset_500]
    mock_manager.pf.plan_current = "basic_plan"  # No match
    result = mock_manager._execute_primary_plan(mock_manager)
    assert result == asset_100


def test_all_assets_are_encroached(mock_manager) -> None:
    """Multiple plans exist but all are encroached. Returns first plan asset."""
    asset_100 = create_mock_asset("Plan_100yr", is_encroached=True)
    asset_500 = create_mock_asset("Plan_500yr", is_encroached=True)
    mock_manager.plan_assets = [asset_100, asset_500]
    result = mock_manager._execute_primary_plan(mock_manager)
    assert result == asset_100


def test_exactly_one_candidate_plan(mock_manager) -> None:
    """Multiple plans exist, but exactly one unencroached plan. Returns it."""
    asset_100 = create_mock_asset("Plan_100yr", is_encroached=True)
    asset_500 = create_mock_asset("Plan_500yr", is_encroached=False)
    mock_manager.plan_assets = [asset_100, asset_500]
    result = mock_manager._execute_primary_plan(mock_manager)
    assert result == asset_500
