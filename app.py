
import os
import json
import hmac
import secrets
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from functools import wraps

import numpy as np
import pandas as pd
import shap

from flask import (
    Flask, jsonify, request, render_template, redirect,
    url_for, session, flash, g, abort
)
from werkzeug.security import generate_password_hash, check_password_hash
from xgboost import XGBClassifier


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_model.json"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
DATABASE_PATH = BASE_DIR / "transactions.db"
SHAP_IMPORTANCE_PATH = (
    BASE_DIR / "model_results" / "shap_feature_importance.csv"
)

app = Flask(__name__)

app.config.update(
    SECRET_KEY=os.environ.get("SECUREPAY_SECRET_KEY")
    or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=16 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=(
        os.environ.get("SECUREPAY_HTTPS", "false").lower() == "true"
    ),
    PERMANENT_SESSION_LIFETIME=60 * 60 * 8,
)


# ==========================================================
# 2. LOAD TRAINED MODEL
# ==========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not METADATA_PATH.exists():
    raise FileNotFoundError(f"Model metadata not found: {METADATA_PATH}")

model = XGBClassifier()
model.load_model(str(MODEL_PATH))

with open(METADATA_PATH, encoding="utf-8") as file:
    metadata = json.load(file)

FEATURE_COLUMNS = metadata.get("feature_columns")

if not isinstance(FEATURE_COLUMNS, list) or not FEATURE_COLUMNS:
    raise ValueError(
        "model_metadata.json must contain feature_columns."
    )

THRESHOLD = float(metadata.get("selected_threshold", 0.5))

if not 0 < THRESHOLD < 1:
    raise ValueError("The model threshold must be between 0 and 1.")

# The metadata describes a binary XGBoost model.
if len(FEATURE_COLUMNS) != 11:
    app.logger.warning(
        "Expected 11 PaySim features; metadata contains %s.",
        len(FEATURE_COLUMNS)
    )

explainer = shap.TreeExplainer(model)


# ==========================================================
# 3. DATABASE
# ==========================================================

def get_connection():
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def column_exists(connection, table, column):
    rows = connection.execute(
        f"PRAGMA table_info({table})"
    ).fetchall()
    return any(row["name"] == column for row in rows)


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user'
                    CHECK (role IN ('admin', 'user')),
                created_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
                    CHECK (is_active IN (0, 1))
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                amount REAL NOT NULL,
                transaction_type TEXT NOT NULL,
                step INTEGER NOT NULL,
                oldbalance_org REAL NOT NULL,
                newbalance_org REAL NOT NULL,
                oldbalance_dest REAL NOT NULL,
                newbalance_dest REAL NOT NULL,
                fraud_probability REAL NOT NULL,
                threshold REAL NOT NULL,
                prediction TEXT NOT NULL,
                created_at TEXT NOT NULL,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL
            )
        """)

        if not column_exists(connection, "transactions", "user_id"):
            connection.execute("""
                ALTER TABLE transactions
                ADD COLUMN user_id INTEGER REFERENCES users(id)
                    ON DELETE SET NULL
            """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_user_id
            ON transactions(user_id)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_created_at
            ON transactions(created_at)
        """)


initialize_database()


# ==========================================================
# 4. INITIAL ADMIN
# ==========================================================

def initialize_admin():
    username = os.environ.get(
        "SECUREPAY_ADMIN_USERNAME", ""
    ).strip()
    email = os.environ.get(
        "SECUREPAY_ADMIN_EMAIL", ""
    ).strip().lower()
    password = os.environ.get("SECUREPAY_ADMIN_PASSWORD", "")

    if not any([username, email, password]):
        app.logger.warning(
            "Admin not created. Configure SECUREPAY_ADMIN_USERNAME, "
            "SECUREPAY_ADMIN_EMAIL and SECUREPAY_ADMIN_PASSWORD."
        )
        return

    if not all([username, email, password]):
        raise RuntimeError(
            "All three SECUREPAY_ADMIN environment variables are required."
        )

    if len(username) < 3 or len(password) < 12:
        raise RuntimeError(
            "Admin username must have 3+ characters and password "
            "must have at least 12 characters."
        )

    with get_connection() as connection:
        existing = connection.execute("""
            SELECT id FROM users
            WHERE username = ? OR email = ?
        """, (username, email)).fetchone()

        if existing:
            app.logger.warning(
                "Admin bootstrap skipped: username or email already exists."
            )
            return

        connection.execute("""
            INSERT INTO users (
                username, email, password_hash, role,
                created_at, is_active
            )
            VALUES (?, ?, ?, 'admin', ?, 1)
        """, (
            username,
            email,
            generate_password_hash(password),
            datetime.now(timezone.utc).isoformat()
        ))


initialize_admin()


# ==========================================================
# 5. AUTHENTICATION AND CSRF
# ==========================================================

def current_user():
    if hasattr(g, "current_user"):
        return g.current_user

    user_id = session.get("user_id")

    if not user_id:
        g.current_user = None
        return None

    with get_connection() as connection:
        user = connection.execute("""
            SELECT id, username, email, role, is_active
            FROM users
            WHERE id = ?
        """, (user_id,)).fetchone()

    if user is None or not user["is_active"]:
        session.clear()
        g.current_user = None
        return None

    g.current_user = dict(user)
    return g.current_user


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        user = current_user()

        if user is None:
            if request.path.startswith("/api/"):
                return jsonify({
                    "error": "Please log in to continue.",
                    "login_required": True
                }), 401

            return redirect(url_for(
                "login_page",
                next=request.full_path.rstrip("?")
            ))

        return view(*args, **kwargs)

    return wrapped_view


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped_view(*args, **kwargs):
        user = current_user()

        if user["role"] != "admin":
            if request.path.startswith("/api/"):
                return jsonify({
                    "error": "Administrator access is required."
                }), 403
            abort(403)

        return view(*args, **kwargs)

    return wrapped_view


@app.context_processor
def inject_template_values():
    if "_csrf_token" not in session:
        session["_csrf_token"] = secrets.token_urlsafe(32)

    # csrf_token is a STRING, so templates must use {{ csrf_token }}.
    return {
        "csrf_token": session["_csrf_token"],
        "logged_in_user": current_user()
    }


def validate_csrf():
    expected = session.get("_csrf_token", "")
    supplied = request.form.get("csrf_token", "")

    if (
        not expected
        or not supplied
        or not hmac.compare_digest(expected, supplied)
    ):
        abort(400, description="Invalid or missing CSRF token.")


# ==========================================================
# 6. AUTHENTICATION ROUTES
# ==========================================================

@app.route("/login", methods=["GET", "POST"])
def login_page():
    user = current_user()

    if user:
        return redirect(
            url_for("admin_dashboard_page")
            if user["role"] == "admin"
            else url_for("dashboard_page")
        )

    if request.method == "POST":
        validate_csrf()

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        login_type = request.form.get("login_type", "user").strip().lower()

        if login_type not in {"user", "admin"}:
            flash("Invalid login type.", "error")
            return render_template("login.html"), 400

        with get_connection() as connection:
            user = connection.execute("""
                SELECT *
                FROM users
                WHERE username = ? OR email = ?
            """, (username, username.lower())).fetchone()

        valid_login = (
            user is not None
            and user["is_active"]
            and check_password_hash(user["password_hash"], password)
        )

        if not valid_login:
            flash("Invalid username/email or password.", "error")
            return render_template("login.html"), 401

        if user["role"] != login_type:
            flash(
                "This account cannot use the selected login portal.",
                "error"
            )
            return render_template("login.html"), 403

        session.clear()
        session["user_id"] = user["id"]
        session.permanent = True
        session["_csrf_token"] = secrets.token_urlsafe(32)

        destination = request.args.get("next", "")

        if (
            destination.startswith("/")
            and not destination.startswith("//")
            and "\\" not in destination
        ):
            return redirect(destination)

        return redirect(
            url_for("admin_dashboard_page")
            if user["role"] == "admin"
            else url_for("dashboard_page")
        )

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register_page():
    if current_user():
        return redirect(url_for("dashboard_page"))

    if request.method == "POST":
        validate_csrf()

        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not 3 <= len(username) <= 40:
            flash("Username must contain 3 to 40 characters.", "error")
            return render_template("register.html"), 400

        if (
            not email
            or len(email) > 254
            or "@" not in email
            or email.startswith("@")
            or email.endswith("@")
        ):
            flash("Enter a valid email address.", "error")
            return render_template("register.html"), 400

        if len(password) < 10 or len(password) > 256:
            flash("Password must contain 10 to 256 characters.", "error")
            return render_template("register.html"), 400

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("register.html"), 400

        try:
            with get_connection() as connection:
                connection.execute("""
                    INSERT INTO users (
                        username, email, password_hash, role,
                        created_at, is_active
                    )
                    VALUES (?, ?, ?, 'user', ?, 1)
                """, (
                    username,
                    email,
                    generate_password_hash(password),
                    datetime.now(timezone.utc).isoformat()
                ))
        except sqlite3.IntegrityError:
            app.logger.exception("Registration database error")
            return (
                "Username or email is already registered. "
                "Please use different registration details.",
                409
            )

        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("login_page"))

    return render_template("register.html")


@app.post("/logout")
@login_required
def logout_page():
    validate_csrf()
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login_page"))


# ==========================================================
# 7. MODEL FEATURE PREPARATION
# ==========================================================

ALLOWED_TYPES = {
    "CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"
}


def prepare_features(data):
    required = [
        "amount", "transaction_type", "step",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest"
    ]

    missing = [key for key in required if key not in data]

    if missing:
        raise ValueError("Missing fields: " + ", ".join(missing))

    transaction_type = str(data["transaction_type"]).strip().upper()

    if transaction_type not in ALLOWED_TYPES:
        raise ValueError("Unsupported transaction type.")

    try:
        amount = float(data["amount"])
        step_value = float(data["step"])
        old_org = float(data["oldbalanceOrg"])
        new_org = float(data["newbalanceOrig"])
        old_dest = float(data["oldbalanceDest"])
        new_dest = float(data["newbalanceDest"])
    except (TypeError, ValueError):
        raise ValueError("Transaction values must be numeric.")

    values = np.array([
        amount, step_value, old_org, new_org, old_dest, new_dest
    ], dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Values must be finite numbers.")

    if amount <= 0:
        raise ValueError("Amount must be greater than zero.")

    if step_value < 1 or not step_value.is_integer():
        raise ValueError("Step must be a positive integer.")

    if any(value < 0 for value in values[2:]):
        raise ValueError("Balances cannot be negative.")

    if not np.isclose(
        old_org - amount, new_org, rtol=0, atol=0.01
    ):
        raise ValueError(
            "Sender balance is inconsistent with the transaction amount."
        )

    row = {
        "step": int(step_value),
        "amount": amount,
        "oldbalanceOrg": old_org,
        "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest
    }

    for transaction in ALLOWED_TYPES:
        row[f"type_{transaction}"] = int(
            transaction_type == transaction
        )

    features = pd.DataFrame([row]).reindex(
        columns=FEATURE_COLUMNS,
        fill_value=0
    ).astype(float)

    return (
        features, transaction_type, amount, int(step_value),
        old_org, new_org, old_dest, new_dest
    )


# ==========================================================
# 8. PREDICTION AND SHAP
# ==========================================================

def predict_transaction(features):
    probability = float(model.predict_proba(features)[0, 1])

    prediction = (
        "Potential Fraud"
        if probability >= THRESHOLD
        else "Likely Genuine"
    )

    contributions = {}

    try:
        shap_values = np.asarray(explainer.shap_values(features))

        if shap_values.ndim == 3:
            shap_values = shap_values[0, :, -1]
        elif shap_values.ndim == 2:
            shap_values = shap_values[0]

        contributions = {
            feature: round(float(value), 6)
            for feature, value in zip(FEATURE_COLUMNS, shap_values)
        }
    except Exception:
        app.logger.exception("SHAP explanation could not be generated.")

    top_contributions = sorted(
        contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True
    )[:5]

    return {
        "fraud_probability": round(probability, 6),
        "fraud_percentage": round(probability * 100, 2),
        "threshold": THRESHOLD,
        "prediction": prediction,
        "shap_contributions": contributions,
        "top_influential_features": [
            {
                "feature": feature,
                "contribution": value,
                "direction": (
                    "increases fraud score" if value > 0
                    else "decreases fraud score" if value < 0
                    else "neutral"
                )
            }
            for feature, value in top_contributions
        ],
        "disclaimer": (
            "This educational model uses a synthetic mobile-money "
            "dataset and is not connected to live UPI transactions."
        )
    }


def save_transaction(user_id, prepared, result):
    (
        _features, transaction_type, amount, step_value,
        old_org, new_org, old_dest, new_dest
    ) = prepared

    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO transactions (
                user_id, amount, transaction_type, step,
                oldbalance_org, newbalance_org,
                oldbalance_dest, newbalance_dest,
                fraud_probability, threshold, prediction, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, amount, transaction_type, step_value,
            old_org, new_org, old_dest, new_dest,
            result["fraud_probability"], THRESHOLD,
            result["prediction"],
            datetime.now(timezone.utc).isoformat()
        ))

        return cursor.lastrowid


# ==========================================================
# 9. STATISTICS
# ==========================================================

def get_statistics(user_id=None):
    where_clause = ""
    parameters = ()

    if user_id is not None:
        where_clause = "WHERE user_id = ?"
        parameters = (user_id,)

    with get_connection() as connection:
        stats = connection.execute(f"""
            SELECT
                COUNT(*) AS total_transactions,
                COALESCE(SUM(prediction = 'Potential Fraud'), 0)
                    AS suspected_frauds,
                COALESCE(SUM(prediction = 'Likely Genuine'), 0)
                    AS likely_genuine,
                COALESCE(AVG(amount), 0) AS average_amount,
                COALESCE(SUM(amount), 0) AS total_amount
            FROM transactions
            {where_clause}
        """, parameters).fetchone()

        daily_rows = connection.execute(f"""
            SELECT
                DATE(created_at) AS day,
                COUNT(*) AS total,
                COALESCE(SUM(prediction = 'Potential Fraud'), 0)
                    AS suspected_frauds
            FROM transactions
            {where_clause}
            GROUP BY DATE(created_at)
            ORDER BY day
        """, parameters).fetchall()

    return dict(stats), [dict(row) for row in daily_rows]


def enrich_transaction(row):
    transaction = dict(row)
    transaction["transaction_id"] = transaction["id"]
    transaction["risk_score"] = round(
        transaction["fraud_probability"] * 100, 2
    )
    transaction["probability"] = transaction["risk_score"]
    transaction["result"] = transaction["prediction"]
    transaction["type"] = transaction["transaction_type"]
    return transaction


# ==========================================================
# 10. WEBSITE PAGES
# ==========================================================

@app.get("/")
def home():
    user = current_user()

    if user:
        return redirect(
            url_for("admin_dashboard_page")
            if user["role"] == "admin"
            else url_for("dashboard_page")
        )

    return render_template("home.html", active_page="home")




@app.get("/dashboard")
@login_required
def dashboard_page():
    user = current_user()

    if user["role"] == "admin":
        return redirect(url_for("admin_dashboard_page"))

    stats, daily = get_statistics(user["id"])

    with get_connection() as connection:
        rows = connection.execute("""
            SELECT *
            FROM transactions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 5
        """, (user["id"],)).fetchall()

    recent = [enrich_transaction(row) for row in rows]

    total = stats["total_transactions"]
    fraud_count = stats["suspected_frauds"]

    return render_template(
        "dashboard.html",
        active_page="dashboard",
        total_transactions=total,
        fraud_count=fraud_count,
        legitimate_count=stats["likely_genuine"],
        fraud_rate=round(fraud_count / total * 100, 2) if total else 0,
        recent_transactions=recent,
        statistics=stats,
        daily_statistics=daily
    )



@app.get("/admin/dashboard")
@admin_required
def admin_dashboard_page():
    stats, daily = get_statistics()

    with get_connection() as connection:
        users_count = connection.execute("""
            SELECT COUNT(*) FROM users WHERE role = 'user'
        """).fetchone()[0]

        rows = connection.execute("""
            SELECT *
            FROM transactions
            ORDER BY id DESC
            LIMIT 10
        """).fetchall()

    return render_template(
        "admin_dashboard.html",
        active_page="admin_dashboard",
        total_users=users_count,
        total_transactions=stats["total_transactions"],
        fraud_count=stats["suspected_frauds"],
        legitimate_count=stats["likely_genuine"],
        model_name=metadata.get("algorithm", "XGBoost"),
        recent_transactions=[enrich_transaction(row) for row in rows],
        statistics=stats,
        daily_statistics=daily
    )


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict_page():
    if current_user()["role"] == "admin":
        return redirect(url_for("admin_dashboard_page"))

    if request.method == "POST":
        validate_csrf()

        try:
            prepared = prepare_features(request.form.to_dict())
            result = predict_transaction(prepared[0])

            transaction_id = save_transaction(
                current_user()["id"], prepared, result
            )

            return render_template(
                "predict.html",
                active_page="predict",
                prediction=result["prediction"],
                probability=result["fraud_percentage"],
                transaction_id=transaction_id,
                result_message=result["disclaimer"],
                result=result
            )

        except ValueError as error:
            flash(str(error), "error")
            return render_template(
                "predict.html", active_page="predict"
            ), 400

        except Exception:
            app.logger.exception("Form prediction failed.")
            flash("Prediction failed. Check the server logs.", "error")
            return render_template(
                "predict.html", active_page="predict"
            ), 500

    return render_template("predict.html", active_page="predict")



@app.get("/history")
@login_required
def history_page():
    user = current_user()

    with get_connection() as connection:
        if user["role"] == "admin":
            rows = connection.execute("""
                SELECT *
                FROM transactions
                ORDER BY id DESC
                LIMIT 100
            """).fetchall()
        else:
            rows = connection.execute("""
                SELECT *
                FROM transactions
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT 100
            """, (user["id"],)).fetchall()

    transactions = []

    for index, row in enumerate(rows, start=1):
        transaction = enrich_transaction(row)

        # Display a serial number without changing the database ID.
        transaction["display_number"] = index

        transactions.append(transaction)

    fraud_count = sum(
        transaction["prediction"] == "Potential Fraud"
        for transaction in transactions
    )

    legitimate_count = sum(
        transaction["prediction"] == "Likely Genuine"
        for transaction in transactions
    )

    return render_template(
        "history.html",
        active_page="history",
        transactions=transactions,
        fraud_count=fraud_count,
        legitimate_count=legitimate_count
    )




@app.get("/model-performance")
@admin_required
def model_performance_page():
    metrics = metadata.get("test_metrics", {})

    return render_template(
        "model_performance.html",
        active_page="model_performance",
        model_name=metadata.get("algorithm", "XGBoost"),
        algorithm=metadata.get("algorithm", "XGBoost"),
        threshold=THRESHOLD,
        accuracy=metrics.get("accuracy", 0),
        precision=metrics.get("precision", 0),
        recall=metrics.get("recall", 0),
        f1_score=metrics.get("f1_score", 0),
        roc_auc=metrics.get("roc_auc", 0),
        average_precision=metrics.get("average_precision", 0),
        training_samples=metadata.get("training_rows", 0),
        validation_samples=metadata.get("validation_rows", 0),
        test_samples=metadata.get("testing_rows", 0),
        training_time=metadata.get("training_time_seconds", 0),
        test_metrics=metrics,
        feature_columns=FEATURE_COLUMNS,
        dataset=metadata.get("dataset", "Synthetic mobile-money dataset"),
        last_updated=metadata.get("last_updated", "Not available")
    )



@app.get("/transaction/<int:transaction_id>")
@login_required
def transaction_details_page(transaction_id):
    user = current_user()

    with get_connection() as connection:
        if user["role"] == "admin":
            row = connection.execute("""
                SELECT * FROM transactions WHERE id = ?
            """, (transaction_id,)).fetchone()
        else:
            row = connection.execute("""
                SELECT *
                FROM transactions
                WHERE id = ? AND user_id = ?
            """, (transaction_id, user["id"])).fetchone()

    if row is None:
        abort(404)

    return render_template(
        "transaction_details.html",
        active_page="history",
        transaction=enrich_transaction(row),
        transaction_id=transaction_id
    )


# ==========================================================
# 11. HEALTH AND PREDICTION API
# ==========================================================

@app.get("/api/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": True})


@app.post("/api/predict")
@login_required
def predict_api():
    if current_user()["role"] != "user":
        return jsonify({
            "error": "Use the administrator dashboard for analytics."
        }), 403

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send transaction details as JSON."}), 400

    try:
        prepared = prepare_features(data)
        result = predict_transaction(prepared[0])
        transaction_id = save_transaction(
            current_user()["id"], prepared, result
        )

        result.update({
            "transaction_id": transaction_id,
            "transaction_type": prepared[1],
            "amount": prepared[2],
            "step": prepared[3]
        })

        return jsonify(result)

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        app.logger.exception("API prediction failed.")
        return jsonify({"error": "Prediction failed."}), 500


# ==========================================================
# 12. HISTORY AND TRANSACTION DETAILS API
# ==========================================================

@app.get("/api/history")
@login_required
def history_api():
    limit = request.args.get("limit", default=50, type=int)

    if limit is None or not 1 <= limit <= 100:
        return jsonify({"error": "Limit must be between 1 and 100."}), 400

    user = current_user()

    with get_connection() as connection:
        if user["role"] == "admin":
            rows = connection.execute("""
                SELECT id, user_id, amount, transaction_type, step,
                       fraud_probability, threshold, prediction, created_at
                FROM transactions
                ORDER BY id DESC LIMIT ?
            """, (limit,)).fetchall()
        else:
            rows = connection.execute("""
                SELECT id, amount, transaction_type, step,
                       fraud_probability, threshold, prediction, created_at
                FROM transactions
                WHERE user_id = ?
                ORDER BY id DESC LIMIT ?
            """, (user["id"], limit)).fetchall()

    return jsonify({"transactions": [dict(row) for row in rows]})


@app.get("/api/history/<int:transaction_id>")
@login_required
def transaction_details_api(transaction_id):
    user = current_user()

    with get_connection() as connection:
        if user["role"] == "admin":
            row = connection.execute("""
                SELECT * FROM transactions WHERE id = ?
            """, (transaction_id,)).fetchone()
        else:
            row = connection.execute("""
                SELECT *
                FROM transactions
                WHERE id = ? AND user_id = ?
            """, (transaction_id, user["id"])).fetchone()

    if row is None:
        return jsonify({"error": "Transaction not found."}), 404

    transaction = dict(row)

    input_data = {
        "amount": transaction["amount"],
        "transaction_type": transaction["transaction_type"],
        "step": transaction["step"],
        "oldbalanceOrg": transaction["oldbalance_org"],
        "newbalanceOrig": transaction["newbalance_org"],
        "oldbalanceDest": transaction["oldbalance_dest"],
        "newbalanceDest": transaction["newbalance_dest"]
    }

    try:
        features, *_ = prepare_features(input_data)
        explanation = predict_transaction(features)
        transaction["shap_contributions"] = explanation["shap_contributions"]
        transaction["top_influential_features"] = (
            explanation["top_influential_features"]
        )
    except Exception:
        app.logger.exception("Could not generate transaction explanation.")
        transaction["shap_contributions"] = {}
        transaction["top_influential_features"] = []

    return jsonify(transaction)


# ==========================================================
# 13. DASHBOARD API
# ==========================================================

@app.get("/api/dashboard")
@login_required
def dashboard_api():
    user = current_user()

    if user["role"] == "admin":
        return jsonify({"error": "Use the admin dashboard."}), 403

    stats, daily = get_statistics(user["id"])

    return jsonify({
        "statistics": stats,
        "daily_statistics": daily
    })


@app.get("/api/admin/overview")
@admin_required
def admin_overview_api():
    stats, daily = get_statistics()

    with get_connection() as connection:
        recent = connection.execute("""
            SELECT id, user_id, amount, transaction_type, step,
                   fraud_probability, prediction, created_at
            FROM transactions
            ORDER BY id DESC LIMIT 10
        """).fetchall()

        users_count = connection.execute("""
            SELECT COUNT(*) FROM users WHERE role = 'user'
        """).fetchone()[0]

    return jsonify({
        "statistics": stats,
        "total_users": users_count,
        "daily_statistics": daily,
        "recent_transactions": [dict(row) for row in recent]
    })


# ==========================================================
# 14. MODEL PERFORMANCE AND FEATURE IMPORTANCE API
# ==========================================================

@app.get("/api/model-performance")
@admin_required
def model_performance_api():
    return jsonify({
        "algorithm": metadata.get("algorithm", "XGBoost"),
        "threshold": THRESHOLD,
        "test_metrics": metadata.get("test_metrics", {}),
        "feature_columns": FEATURE_COLUMNS,
        "dataset": metadata.get(
            "dataset", "Synthetic mobile-money dataset"
        ),
        "note": (
            "Results apply to the held-out test set and do not "
            "establish real-world UPI performance."
        )
    })


@app.get("/api/feature-importance")
@admin_required
def feature_importance_api():
    if not SHAP_IMPORTANCE_PATH.exists():
        return jsonify({
            "error": "SHAP importance file is missing. "
                     "Run the model explanation script first."
        }), 404

    try:
        importance = pd.read_csv(SHAP_IMPORTANCE_PATH).head(10)
        return jsonify({
            "method": "SHAP",
            "features": importance.to_dict(orient="records")
        })
    except Exception:
        app.logger.exception("Failed to read feature importance.")
        return jsonify({"error": "Could not load feature importance."}), 500


# ==========================================================
# 15. ERROR HANDLERS
# ==========================================================

@app.errorhandler(400)
def bad_request(error):
    if request.path.startswith("/api/"):
        return jsonify({
            "error": getattr(error, "description", "Bad request.")
        }), 400

    return render_template(
        "error.html",
        error_code=400,
        error_message=getattr(error, "description", "Bad request.")
    ), 400


@app.errorhandler(403)
def forbidden(error):
    if request.path.startswith("/api/"):
        return jsonify({
            "error": "You do not have permission to access this resource."
        }), 403

    return render_template(
        "error.html",
        error_code=403,
        error_message="You do not have permission to access this page."
    ), 403


@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Page or API endpoint not found."}), 404

    return render_template(
        "error.html",
        error_code=404,
        error_message="The requested page could not be found."
    ), 404


@app.errorhandler(413)
def request_too_large(error):
    return jsonify({"error": "Request is too large."}), 413


# ==========================================================
# 16. START APPLICATION
# ==========================================================

if __name__ == "__main__":
    print("=" * 55)
    print("SECUREPAY - ML FRAUD DETECTION SYSTEM")
    print("=" * 55)
    print("Model: XGBoost")
    print(f"Threshold: {THRESHOLD:.6f}")
    print("Website: http://127.0.0.1:5000")
    print("=" * 55)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
