def get_slots():
    """Return the current parking slot occupancy data."""
    return [
        {"id": "S1", "occupied": False},
        {"id": "S2", "occupied": True},
        {"id": "S3", "occupied": False},
        {"id": "S4", "occupied": True},
    ]


def allocate_slot(car_id):
    """Return a simple allocation result for the given car.

    The car_id is accepted now so this function can later use the trained
    SmartParking-RL policy for real slot allocation.
    """
    return {
        "allocated_slot": "S4",
        "occupancy": 4,
        "reward": 12,
        "status": "allocated",
    }


def get_metrics():
    """Return basic SmartParking metrics for the dashboard/API."""
    return {
        "occupancy": 4,
        "free_slots": 6,
        "reward": 12,
        "cars_processed": 20,
    }
