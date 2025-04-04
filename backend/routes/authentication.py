from flask import Blueprint, request, jsonify
import os
import json
import uuid

auth_routes = Blueprint('auth_routes', __name__)
USERS_FILE = 'storage/users.json'

# Load users from JSON
def load_users():
    if not os.path.exists(USERS_FILE):
        return {"users": []}
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

# Save users to JSON
def save_users(users_data):
    with open(USERS_FILE, 'w') as f:
        json.dump(users_data, f, indent=4)

# -------------------------
# 🔐 Register Endpoint
# -------------------------
@auth_routes.route('/api/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name')
    role = data.get('role')  # "student" or "teacher"

    if not name or not role:
        return jsonify({"error": "Missing name or role"}), 400

    users_data = load_users()

    # Generate unique ID
    user_id = str(uuid.uuid4())

    new_user = {
        "user_id": user_id,
        "name": name,
        "role": role
    }

    users_data["users"].append(new_user)
    save_users(users_data)

    return jsonify({"message": "User registered successfully", "user": new_user}), 201

# -------------------------
# 🔓 Login Endpoint
# -------------------------
@auth_routes.route('/api/login', methods=['POST'])
def login():
    data = request.json
    name = data.get('name')

    if not name:
        return jsonify({"error": "Missing name"}), 400

    users_data = load_users()
    for user in users_data["users"]:
        if user["name"] == name:
            return jsonify({"message": "Login successful", "user": user}), 200

    return jsonify({"error": "User not found"}), 404
