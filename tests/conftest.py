import pytest


@pytest.fixture
def sample_user_data():
    return {"email": "test@example.com", "password": "SecurePass123!"}


@pytest.fixture
def sample_prediction_input():
    return {
        "rooms": 2,
        "area": 50.0,
        "floor": 5,
        "total_floors": 10,
        "year_built": 2005,
        "distance_to_metro_min": 10,
    }
