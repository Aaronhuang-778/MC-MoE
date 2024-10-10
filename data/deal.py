import pandas as pd

df = pd.read_parquet("train-00000-of-00001.parquet")
json_str = df.to_json(orient='records')
with open('train-00000-of-00001.json', 'w') as f:
    f.write(json_str)
