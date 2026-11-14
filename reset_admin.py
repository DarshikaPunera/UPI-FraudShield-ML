import sqlite3
import os
from werkzeug.security import generate_password_hash

DATABASE_PATH = "transactions.db"

username = "admin"
email = "admin@securepay.local"
new_password = os.environ.get("SECUREPAY_ADMIN_PASSWORD")

if not new_password:
    raise SystemExit(
        "Please set SECUREPAY_ADMIN_PASSWORD in the terminal first."
    )

connection = sqlite3.connect(DATABASE_PATH)

try:
    user = connection.execute(
        """
        SELECT id, username, email, role
        FROM users
        WHERE username = ? OR email = ?
        """,
        (username, email)
    ).fetchone()

    if user:
        connection.execute(
            """
            UPDATE users
            SET password_hash = ?, role = 'admin', is_active = 1
            WHERE id = ?
            """,
            (generate_password_hash(new_password), user[0])
        )
        print("Existing admin account updated successfully.")

    else:
        connection.execute(
            """
            INSERT INTO users
            (username, email, password_hash, role, created_at, is_active)
            VALUES (?, ?, ?, 'admin', datetime('now'), 1)
            """,
            (
                username,
                email,
                generate_password_hash(new_password)
            )
        )
        print("New admin account created successfully.")

    connection.commit()

except Exception:
    connection.rollback()
    raise

finally:
    connection.close()