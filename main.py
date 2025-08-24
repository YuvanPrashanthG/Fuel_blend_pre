from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import pandas as pd
import joblib
import io

app = FastAPI()

# Load models
models = {}
for k in range(1, 11):
    models[f"BlendProperty{k}"] = joblib.load(f"models/BlendProperty{k}.pkl")

@app.post("/predict-csv/")
async def predict_csv(file: UploadFile = File(...)):
    # Read uploaded CSV file into pandas DataFrame
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    results = []

    # Loop through each row in the CSV
    for idx, row in df.iterrows():
        row_preds = {}
        for k in range(1, 11):
            frac_cols = [f"Component{i}_fraction" for i in range(1, 6)]
            prop_cols = [f"Component{i}_Property{k}" for i in range(1, 6)]
            X_sample = row[frac_cols + prop_cols].values.reshape(1, -1)
            row_preds[f"BlendProperty{k}"] = models[f"BlendProperty{k}"].predict(X_sample)[0]
        results.append(row_preds)

    # Convert results into DataFrame
    result_df = pd.DataFrame(results)

    # Convert DataFrame to CSV
    output = io.StringIO()
    result_df.to_csv(output, index=False)
    output.seek(0)

    # Return as downloadable CSV
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"}
    )
