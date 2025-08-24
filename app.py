import os
import pandas as pd
import mysql.connector
from flask import Flask, request, render_template, send_file, redirect, url_for, flash
import joblib
import io
from datetime import datetime
import traceback
import numpy as np

app = Flask(__name__, template_folder="templates")
app.secret_key = 'fuel_blend_secret_key'

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'nithi@2005',
    'database': 'fuel_blend_db'
}

# Load the trained model
try:
    model = joblib.load("xgb_model.pkl")
    print("✅ Model loaded successfully")
    print(f"✅ Model type: {type(model)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(traceback.format_exc())
    model = None

def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as e:
        print(f"❌ Database connection error: {e}")
        flash(f"Database error: {e}")
        return None

def init_db():
    try:
        conn = get_db_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_files (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_content LONGBLOB,
                status VARCHAR(20) DEFAULT 'uploaded'
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                file_id INT,
                prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                prediction_data LONGTEXT,
                FOREIGN KEY (file_id) REFERENCES uploaded_files(id) ON DELETE CASCADE
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Database initialized successfully")
        return True
        
    except mysql.connector.Error as e:
        print(f"❌ Database initialization error: {e}")
        return False

def process_file(file_id, file_content):
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Database connection failed", None
            
        cursor = conn.cursor()
        
        # Update status to processing
        cursor.execute("UPDATE uploaded_files SET status = 'processing' WHERE id = %s", (file_id,))
        conn.commit()
        
        # Read and process file
        df = pd.read_csv(io.BytesIO(file_content))
        print(f"📊 Read CSV with shape: {df.shape}")
        print(f"📊 DataFrame columns: {df.columns.tolist()}")
        print(f"📊 First few rows:\n{df.head()}")
        
        # Make predictions
        predictions = model.predict(df)
        print(f"🤖 Predictions shape: {predictions.shape}")
        print(f"🤖 Predictions type: {type(predictions)}")
        print(f"🤖 First few predictions: {predictions[:5]}")
        
        # Create result DataFrame
        result_df = pd.DataFrame()
        result_df['ID'] = range(1, len(predictions) + 1)
        
        # Handle different prediction output formats
        if predictions.ndim == 1:
            # Single column predictions
            result_df['Prediction'] = predictions
        elif predictions.ndim == 2:
            # Multiple column predictions
            for i in range(predictions.shape[1]):
                result_df[f'Prediction_{i+1}'] = predictions[:, i]
        else:
            raise ValueError(f"Unexpected predictions shape: {predictions.shape}")
        
        print(f"📝 Result DataFrame shape: {result_df.shape}")
        print(f"📝 Result DataFrame:\n{result_df.head()}")
        
        # Convert results to CSV string
        results_csv = result_df.to_csv(index=False)
        print(f"💾 CSV data length: {len(results_csv)} characters")
        print(f"💾 CSV data preview:\n{results_csv[:200]}...")
        
        # Store predictions in database
        cursor.execute(
            "INSERT INTO predictions (file_id, prediction_data) VALUES (%s, %s)",
            (file_id, results_csv)
        )
        
        # Update status to completed
        cursor.execute("UPDATE uploaded_files SET status = 'completed' WHERE id = %s", (file_id,))
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ File processed successfully")
        return True, "File processed successfully", result_df
        
    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        
        # Update file status to error
        try:
            if conn:
                cursor.execute(
                    "UPDATE uploaded_files SET status = 'error' WHERE id = %s",
                    (file_id,)
                )
                conn.commit()
                cursor.close()
                conn.close()
        except:
            pass
        
        return False, error_msg, None

# Routes
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if not file.filename.endswith('.csv'):
        flash('Only CSV files are allowed')
        return redirect(request.url)
    
    if not model:
        flash('ML model is not loaded')
        return redirect(request.url)
    
    try:
        file_content = file.read()
        
        conn = get_db_connection()
        if not conn:
            flash('Database connection failed')
            return redirect(request.url)
            
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO uploaded_files (filename, file_content, status) VALUES (%s, %s, %s)",
            (file.filename, file_content, 'uploaded')
        )
        file_id = cursor.lastrowid
        conn.commit()
        cursor.close()
        conn.close()
        
        # Process file
        success, message, result_df = process_file(file_id, file_content)
        
        if success:
            # Save to downloads folder
            os.makedirs("downloads", exist_ok=True)
            output_filename = f"predictions_{file_id}.csv"
            output_path = os.path.join("downloads", output_filename)
            result_df.to_csv(output_path, index=False)
            
            flash('File processed successfully!')
            return redirect(url_for('results', file_id=file_id))
        else:
            flash(message)
            return redirect(request.url)
            
    except Exception as e:
        flash(f'Error: {str(e)}')
        return redirect(request.url)

@app.route('/results/<int:file_id>')
def results(file_id):
    try:
        conn = get_db_connection()
        if not conn:
            flash('Database connection failed')
            return redirect(url_for('index'))
            
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT * FROM uploaded_files WHERE id = %s", (file_id,))
        file_info = cursor.fetchone()
        
        cursor.execute("SELECT * FROM predictions WHERE file_id = %s", (file_id,))
        prediction = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if not file_info:
            flash('File not found')
            return redirect(url_for('index'))
        
        prediction_sample = ""
        if prediction and prediction.get('prediction_data'):
            prediction_sample = '\n'.join(prediction['prediction_data'].split('\n')[:6])
        
        return render_template('results.html', 
                             file_info=file_info, 
                             prediction=prediction,
                             prediction_sample=prediction_sample,
                             file_id=file_id)
                              
    except Exception as e:
        flash(f"Error retrieving results: {str(e)}")
        return redirect(url_for('index'))

@app.route('/history')
def history():
    try:
        conn = get_db_connection()
        if not conn:
            flash('Database connection failed')
            return redirect(url_for('index'))
            
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT uf.id, uf.filename, uf.upload_time, uf.status, 
                   p.prediction_time 
            FROM uploaded_files uf
            LEFT JOIN predictions p ON uf.id = p.file_id
            ORDER BY uf.upload_time DESC
        """)
        
        files = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return render_template('history.html', files=files)
        
    except Exception as e:
        flash(f"Error retrieving history: {str(e)}")
        return redirect(url_for('index'))

@app.route('/download/<int:file_id>')
def download_file(file_id):
    try:
        conn = get_db_connection()
        if not conn:
            flash('Database connection failed')
            return redirect(url_for('history'))
            
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT prediction_data FROM predictions WHERE file_id = %s", (file_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result and result.get('prediction_data'):
            return send_file(
                io.BytesIO(result['prediction_data'].encode()),
                as_attachment=True,
                download_name=f"predictions_{file_id}.csv",
                mimetype='text/csv'
            )
        
        flash("Prediction data not found")
        return redirect(url_for('history'))
    
    except Exception as e:
        flash(f"Error downloading file: {str(e)}")
        return redirect(url_for('history'))

# Debug routes
@app.route('/debug/model')
def debug_model():
    """Debug endpoint to check model status"""
    try:
        if model is None:
            return "❌ Model is not loaded"
        
        model_info = {
            'type': str(type(model)),
            'loaded': model is not None,
        }
        
        # Try to get model parameters if available
        try:
            if hasattr(model, 'get_params'):
                model_info['params'] = model.get_params()
        except:
            model_info['params'] = 'Not available'
        
        return f"""
        <h1>Model Debug Info</h1>
        <pre>{model_info}</pre>
        <a href="/">Back to upload</a>
        """
    except Exception as e:
        return f"❌ Debug error: {str(e)}"

@app.route('/debug/test-prediction')
def debug_test_prediction():
    """Test prediction with sample data"""
    try:
        if model is None:
            return "❌ Model is not loaded"
        
        # Create sample data (adjust based on your model's expected features)
        sample_data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [0.5, 1.5, 2.5, 3.5, 4.5],
            'feature3': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        print(f"🧪 Sample data:\n{sample_data}")
        
        # Make prediction
        predictions = model.predict(sample_data)
        
        result = {
            'sample_data': sample_data.to_dict(),
            'predictions_shape': predictions.shape,
            'predictions_type': str(type(predictions)),
            'predictions_values': predictions.tolist() if hasattr(predictions, 'tolist') else str(predictions)
        }
        
        return f"""
        <h1>Test Prediction</h1>
        <pre>{result}</pre>
        <a href="/">Back to upload</a>
        """
    except Exception as e:
        return f"❌ Test prediction error: {str(e)}<br>{traceback.format_exc()}"

@app.route('/debug/database')
def debug_database():
    """Debug database content"""
    try:
        conn = get_db_connection()
        if not conn:
            return "❌ Database connection failed"
            
        cursor = conn.cursor(dictionary=True)
        
        # Check uploaded_files
        cursor.execute("SELECT * FROM uploaded_files ORDER BY id DESC LIMIT 5")
        uploaded_files = cursor.fetchall()
        
        # Check predictions
        cursor.execute("SELECT id, file_id, LENGTH(prediction_data) as data_length, LEFT(prediction_data, 100) as data_preview FROM predictions ORDER BY id DESC LIMIT 5")
        predictions = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return f"""
        <h1>Database Debug</h1>
        <h2>Uploaded Files (last 5):</h2>
        <pre>{uploaded_files}</pre>
        
        <h2>Predictions (last 5):</h2>
        <pre>{predictions}</pre>
        
        <a href="/">Back to upload</a>
        """
    except Exception as e:
        return f"❌ Database debug error: {str(e)}"

@app.route('/debug/clear-db')
def debug_clear_db():
    """Clear database for testing"""
    try:
        conn = get_db_connection()
        if not conn:
            return "❌ Database connection failed"
            
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        cursor.execute("DELETE FROM uploaded_files")
        conn.commit()
        cursor.close()
        conn.close()
        
        return "✅ Database cleared successfully. <a href='/'>Back to upload</a>"
    except Exception as e:
        return f"❌ Error clearing database: {str(e)}"

if __name__ == '__main__':
    # Create directories
    os.makedirs("downloads", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    # Initialize database
    init_db()
    
    print("🚀 Server starting on http://localhost:5000")
    print("🔧 Debug endpoints available:")
    print("   - http://localhost:5000/debug/model")
    print("   - http://localhost:5000/debug/test-prediction")
    print("   - http://localhost:5000/debug/database")
    print("   - http://localhost:5000/debug/clear-db")
    
    app.run(debug=True, host='0.0.0.0', port=5000)