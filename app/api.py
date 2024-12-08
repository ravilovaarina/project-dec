from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel

DATA_PATH = "../data/mountains_vs_beaches_preferences.csv"
df = pd.read_csv(DATA_PATH)

app = FastAPI()

@app.get("/data")
def get_data(start: int = 0, limit: int = 10):
    """
    Get the data fom dataset.
    """
    data_subset = df.iloc[start:start + limit]
    return data_subset.to_dict(orient="records")

class ColumnData(BaseModel):
    column_name: str
    values: list

@app.post("/add_column")
def add_column(column_data: ColumnData):
    """
    Add a new column with the given name and values to the dataset.
    """
    global df
    
    if len(column_data.values) != len(df):
        return {"error": "The length of the new column values must match the number of rows in the dataset."}
    
    df[column_data.column_name] = column_data.values

    df.to_csv(DATA_PATH, index=False)
    
    return {
        "message": f"Column '{column_data.column_name}' added successfully!",
        "new_column": column_data.column_name
    }
