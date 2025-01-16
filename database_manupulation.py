import psycopg2
from psycopg2 import sql
import json

# Database connection settings
DB_SETTINGS = {
    "dbname": "gpm_thesis",
    "user": "postgres",
    "password": "ephrem",
    "host": "localhost",
    "port": "1998"
}

# Function to connect to the database
def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_SETTINGS)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

# CREATE: Insert data into asset_lists
def create_asset(asset_model, original_value, non_depreciable_value, book_value, acquisition_date):
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                query = """
                INSERT INTO asset_lists (asset_model, original_value, non_depreciable_value, book_value, acquisition_date)
                VALUES (%s, %s, %s, %s, %s) RETURNING id;
                """
                cur.execute(query, (asset_model, original_value, non_depreciable_value, book_value, acquisition_date))
                asset_id = cur.fetchone()[0]
                conn.commit()
                print(f"Asset created with ID: {asset_id}")
        except Exception as e:
            print(f"Error inserting asset: {e}")
        finally:
            conn.close()

# READ: Fetch all assets
def get_all_assets():
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM asset_lists;")
                assets = cur.fetchall()
                for asset in assets:
                    print(asset)
        except Exception as e:
            print(f"Error fetching assets: {e}")
        finally:
            conn.close()

# UPDATE: Update asset details
def update_asset(asset_id, book_value):
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                query = """
                UPDATE asset_lists SET book_value = %s WHERE id = %s;
                """
                cur.execute(query, (book_value, asset_id))
                conn.commit()
                print(f"Asset ID {asset_id} updated successfully.")
        except Exception as e:
            print(f"Error updating asset: {e}")
        finally:
            conn.close()

# DELETE: Remove an asset
def delete_asset(asset_id):
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                query = "DELETE FROM asset_lists WHERE id = %s;"
                cur.execute(query, (asset_id,))
                conn.commit()
                print(f"Asset ID {asset_id} deleted successfully.")
        except Exception as e:
            print(f"Error deleting asset: {e}")
        finally:
            conn.close()

# CREATE: Insert sensor data
def insert_sensor_data(asset_model_id, sensor_data):
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                query = """
                INSERT INTO sensor_data (asset_model_id, sensor_data, created_date)
                VALUES (%s, %s, NOW()) RETURNING id;
                """
                cur.execute(query, (asset_model_id, json.dumps(sensor_data)))
                sensor_id = cur.fetchone()[0]
                conn.commit()
                print(f"Sensor data inserted with ID: {sensor_id}")
        except Exception as e:
            print(f"Error inserting sensor data: {e}")
        finally:
            conn.close()

# READ: Fetch sensor data for a specific asset
def get_sensor_data(asset_model_id):
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                query = "SELECT * FROM sensor_data WHERE asset_model_id = %s;"
                cur.execute(query, (asset_model_id,))
                sensor_data = cur.fetchall()
                for data in sensor_data:
                    print(data)
        except Exception as e:
            print(f"Error fetching sensor data: {e}")
        finally:
            conn.close()

# DELETE: Remove sensor data for a specific asset
def delete_sensor_data(sensor_id):
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                query = "DELETE FROM sensor_data WHERE id = %s;"
                cur.execute(query, (sensor_id,))
                conn.commit()
                print(f"Sensor data ID {sensor_id} deleted successfully.")
        except Exception as e:
            print(f"Error deleting sensor data: {e}")
        finally:
            conn.close()

# Testing CRUD Operations
if __name__ == "__main__":
    # Insert test data
    create_asset("Excavator", 100000, 5000, 95000, "2022-01-10")
    
    # Fetch all assets
    print("\nAll Assets:")
    get_all_assets()
    
    # Update asset
    update_asset(1, 90000)
    
    # Insert sensor data
    insert_sensor_data(1, {"temperature": 75, "vibration": 0.02, "oil_pressure": 50})
    
    # Fetch sensor data
    print("\nSensor Data for Asset ID 1:")
    get_sensor_data(1)
    
    # Delete sensor data
    delete_sensor_data(1)
    
    # Delete asset
    delete_asset(1)
