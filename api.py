import os
import json
import psycopg2
import pandas as pd
from constant import DB_SETTINGS
import joblib
from flask import Flask, request, jsonify
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from psycopg2 import sql
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Initialize Flask app
app = Flask(__name__)

# Database connection settings

# Directory for saving models
MODEL_DIR = "Models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Function to connect to the database
def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_SETTINGS)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None
@app.route("/")
def home_page():
    return jsonify({"message": "Welcome to the Machine Learning Model API!"})

import os
import json
import psycopg2
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Flask app
app = Flask(__name__)



# Directory to store models
MODEL_DIR = "Models"

# Function to connect to the database
def get_db_connection():
    try:
        return psycopg2.connect(**DB_SETTINGS)
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

# 🔹 Endpoint to train a model
@app.route("/train_model", methods=["POST"])
def train_model():
    try:
        # Get JSON request data
        req_data = request.get_json()

        # Validate required fields
        if not req_data or "data" not in req_data or "machine_model" not in req_data or "target_column" not in req_data:
            return jsonify({"error": "Invalid JSON format. Required keys: data, machine_model, target_column"}), 400

        machine_model = req_data["machine_model"]
        target_column = req_data["target_column"]
        table_data = req_data["data"]
        passed_data_from_api = {}
        print(f" target column: {type(table_data[0])}")
        for input_data in table_data:
            passed_data_from_api[input_data['name']] = input_data["values"]
        new_table_data = passed_data_from_api

        print(f" target column: {target_column}")

        # Convert to DataFrame
        df = pd.DataFrame(new_table_data)
        

        # Validate target column
        if target_column not in df.columns:
            return jsonify({"error": f"Target column '{target_column}' not found in data"}), 400

        # Separate features (X) and target (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        column_names = X.columns
        target_column_name = y.name
        print(X.head())

        # Capture input metadata
        model_input_info = {
            "columns": list(X.columns),
            "shape": X.shape,
            "data_types": {col: str(dtype) for col, dtype in X.dtypes.items()}
        }
        model_output_info = {
            "output_column": target_column,
            "output_type": str(y.dtype)
        }

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LinearRegression()
        print(X_train.shape, len(y_train)) 
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)

        # R squared score
        r2 = r2_score(y_test, y_pred_test)
        print(f"R-squared: {r2}")
        # Mean absolute error
        mse = mean_squared_error(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Squared Error: {mae}")
        # Save model
        model_filename = f"{machine_model}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
        os.makedirs(f"{MODEL_DIR}/{machine_model}", exist_ok=True)
        model_path = os.path.join(f"{MODEL_DIR}/{machine_model}", model_filename)
        joblib.dump(model, model_path)

        # Store metadata in the database
        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    query = """
                    INSERT INTO trained_model (asset_model, model_name, model_directory, model_trained_date, model_input_info, model_output_info)
                    VALUES (%s, %s, %s, %s, %s, %s);
                    """
                    cur.execute(query, (
                        machine_model, 
                        model_filename, 
                        model_path, 
                        datetime.now(), 
                        json.dumps(model_input_info), 
                        json.dumps(model_output_info)
                    ))
                    conn.commit()
            except Exception as e:
                print(f"Database insert error: {e}")
            finally:
                conn.close()

        # 🔹 Perform inference on test data (max 20 samples)
        num_samples = min(20, len(X_test))
        X_test_sample = X_test.iloc[:num_samples]
        y_pred = model.predict(X_test_sample)
        df_dict = X_test_sample.to_dict(orient="records")
        for i in range(len(df_dict)):
            df_dict[i][f"{target_column_name}"] = y_pred[i] 

        # 🔹 Save sensor data (predictions)
        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    for i in range(num_samples):
                        query = """
                            SELECT id FROM asset_lists WHERE asset_model = %s
                        """
                        cur.execute(query, (machine_model,))
                        asset_id = cur.fetchone()
                        # sensor_data = {
                        #     "asset_model_id": asset_id,
                        #     "sensor_data": json.dumps(df_dict[i]),
                        #     "created_date": datetime.now(),
                        # }
                        query = """
                        INSERT INTO sensor_data (asset_model_id, sensor_data, created_date, target_column)
                        VALUES (%s, %s, %s, %s);
                        """
                        cur.execute(query, (
                            asset_id, 
                            json.dumps(df_dict[i]), 
                            datetime.now(),
                            target_column_name
                        ))
                    conn.commit()
            except Exception as e:
                print(f"Sensor data insert error: {e}")
            finally:
                conn.close()

        return jsonify({"message": "Model trained and saved successfully", "model_path": model_path})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 🔹 Endpoint to make predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        req_data = request.get_json()

        if not req_data or "machine_model" not in req_data or "input_data" not in req_data:
            return jsonify({"error": "Invalid JSON format. Required keys: machine_model, input_data"}), 400

        machine_model = req_data["machine_model"]
        input_data = req_data["input_data"]  # Can be a single list or a batch of lists

        # Convert single list input to batch format if necessary
        if isinstance(input_data[0], (int, float)):  
            input_data = [input_data]

        # Fetch model details from database
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Failed to connect to the database"}), 500

        try:
            with conn.cursor() as cur:
                query = """
                SELECT model_name, model_directory, model_input_info, model_output_info 
                FROM trained_model WHERE asset_model = %s ORDER BY model_trained_date DESC LIMIT 1;
                """
                cur.execute(query, (machine_model,))
                model_record = cur.fetchone()

                if not model_record:
                    return jsonify({"error": f"No trained model found for {machine_model}"}), 404

                model_name, model_path, model_input_info, model_output_info = model_record

        finally:
            conn.close()

        # Ensure the model file exists
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model file {model_path} not found"}), 500

        # Load the trained model
        model = joblib.load(model_path)

        # Convert input data to DataFrame
        feature_columns = model_input_info.split(", ")  # Extract expected feature names
        input_df = pd.DataFrame(input_data, columns=feature_columns)

        # Make predictions
        predictions = model.predict(input_df).tolist()

        return jsonify({
            "machine_model": machine_model,
            "model_used": model_name,
            "predicted_values": predictions,
            "output_info": model_output_info
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🔹 Endpoint to get all asset lists
@app.route("/get_assets", methods=["GET"])
def get_assets():
    try:
        # Connect to the database
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Failed to connect to the database"}), 500
        
        assets = []
        try:
            with conn.cursor() as cur:
                query = "SELECT id, asset_model, original_value, acquisition_date, non_depreciable_value, book_value, depreciation_rate, depreciation_period FROM asset_lists"
                cur.execute(query)
                result = cur.fetchall()
                
                # Transform the query result into a list of dictionaries
                for row in result:
                    asset = {
                        "id": row[0],
                        "assetModel": row[1],
                        "originalValue": row[2],
                        "acquisitionDate": row[3].strftime("%Y-%m-%d"),
                        "nonDepreciableValue": row[4],
                        "bookValue": row[5],
                        "depreciationRate": row[6],
                        "depreciationTimeInterval": row[7]
                    }
                    assets.append(asset)
        finally:
            conn.close()

        return jsonify(assets)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
    
# 🔹 Endpoint to get all sensor data
@app.route("/all_sensor_data", methods=["GET"])
def all_sensor_data():
    try:
        # Connect to the database
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Failed to connect to the database"}), 500
        
        sensor_data = []
        try:
            with conn.cursor() as cur:
                query = """
                    SELECT id, asset_model_id, sensor_data, created_date
                    FROM sensor_data
                """
                cur.execute(query)
                result = cur.fetchall()
                
                # Transform the query result into a list of dictionaries
                for row in result:
                    sensor_record = {
                        "id": row[0],
                        "assetModel": row[1],
                        "sensorData": row[2],  # Assuming sensor_data is stored as a JSON-like text field
                        "createdDate": row[3].strftime("%Y-%m-%d %H:%M:%S")
                    }
                    sensor_data.append(sensor_record)
        finally:
            conn.close()

        return jsonify(sensor_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# 🔹 Endpoint to get sensor data by asset_model
@app.route("/sensor_data/asset", methods=["POST"])
def get_sensor_data_by_asset():
    try:
        # Get JSON data from request
        req_data = request.get_json()

        if not req_data or "asset_model" not in req_data:
            return jsonify({"error": "Invalid JSON format. Required key: asset_model"}), 400

        asset_model = req_data["asset_model"]

        # Connect to the database
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Failed to connect to the database"}), 500
        
        try:
            # Fetch the asset ID using the asset_model
            with conn.cursor() as cur:
                query = """
                    SELECT id FROM asset_lists WHERE asset_model = %s
                """
                cur.execute(query, (asset_model,))
                asset_id = cur.fetchone()

                if not asset_id:
                    return jsonify({"status": f"Asset Not Found"})

                # Fetch all sensor data for the asset ID
                query = """
                    SELECT id, asset_model_id, sensor_data, created_date, target_column
                    FROM sensor_data WHERE asset_model_id = %s
                """
                cur.execute(query, (asset_id,))
                result = cur.fetchall()
                if not result:
                    return jsonify({"status": "Asset Not Found"})

                sensor_data = []
                # Transform the query result into a list of dictionaries
                for row in result:
                    sensor_record = {
                        "id": row[0],
                        "assetModel": row[1],
                        "sensorData": row[2],  # Assuming sensor_data is stored as a JSON-like text field
                        "createdDate": row[3].strftime("%Y-%m-%d %H:%M:%S"),
                        "targetColumn": row[4]
                    }
                    sensor_data.append(sensor_record)

        finally:
            conn.close()

        return jsonify(sensor_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🔹 Endpoint to add a new asset
@app.route("/add_assets", methods=["POST"])
def add_asset():
    try:
        # Get JSON data from request
        req_data = request.get_json()

        # Validate input
        required_fields = ["asset_model", "original_value", "acquisition_date", "non_depreciable_value", "book_value", "depreciation_rate", "depreciation_period"]
        if not all(field in req_data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        asset_model = req_data["asset_model"]
        original_value = req_data["original_value"]
        acquisition_date = req_data["acquisition_date"]
        non_depreciable_value = req_data["non_depreciable_value"]
        book_value = req_data["book_value"]
        depreciation_rate = req_data["depreciation_rate"]
        depreciation_period = req_data["depreciation_period"]

        # Connect to the database
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Failed to connect to the database"}), 500

        try:
            with conn.cursor() as cur:
                # Insert new asset into asset_list table
                query = """
                    INSERT INTO asset_lists (asset_model, original_value, acquisition_date, non_depreciable_value, book_value, depreciation_rate, depreciation_period)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """
                cur.execute(query, (asset_model, original_value, acquisition_date, non_depreciable_value, book_value, depreciation_rate, depreciation_period))
                asset_id = cur.fetchone()[0]  # Get the generated asset ID

                # Commit transaction
                conn.commit()
        finally:
            conn.close()

        return jsonify({"message": "Asset added successfully", "asset_id": asset_id}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
