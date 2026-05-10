from flask import Blueprint, jsonify, request

from rl.inference import allocate_slot, get_metrics, get_slots

api_bp = Blueprint("api", __name__)


@api_bp.get("/")
def home():
    return jsonify({"message": "SmartParking Backend Running"})


@api_bp.get("/slots")
def slots():
    return jsonify(get_slots())


@api_bp.post("/allocate")
def allocate():
    data = request.get_json(silent=True) or {}
    car_id = data.get("car_id")

    allocation = allocate_slot(car_id)
    return jsonify(allocation)


@api_bp.get("/metrics")
def metrics():
    return jsonify(get_metrics())
