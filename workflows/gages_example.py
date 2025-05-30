import os
import pyarrow.parquet as pq
import pandas as pd
import pyarrow as pa
import boto3

s3 = boto3.client("s3")
bucket_name = "trinity-pilot"
events = ["dec_1991", "apr_1990", "may_2015", "nov_2015", "oct_2015"]

for event_filename in events:
    event_id = event_filename.replace("_", "")
    existing_key = f"s3://{bucket_name}/stac/prod-support/gages/{event_filename}.parquet"

    df = table = pd.read_parquet(existing_key)
    for col in df.columns:
        gage_id = col.split("_")[2]
        print(event_id, gage_id)

        df_slice = df[[f"gage_usgs_{gage_id}"]].rename(columns={f"gage_usgs_{gage_id}": "flow"})
        new_key = f"s3://{bucket_name}/stac/prod-support/pq-test/gage={gage_id}/event={event_id}/data.pq"
        df_slice.to_parquet(new_key)

# -------------------------------------------

import duckdb

conn = duckdb.connect()
conn.execute("INSTALL 'httpfs';")
conn.execute("LOAD 'httpfs';")

# conn.execute("SET s3_access_key_id='AWS_ACCESS_KEY_ID';")
# conn.execute("SET s3_secret_access_key='S3_SECRET_ACCESS_KEY';")
# conn.execute("SET s3_region='AWS_REGION';")


s3_path = "s3://trinity-pilot/stac/prod-support/pq-test/**/data.pq"

# gage_id = "08045850"
# event_id = "dec1991"

# query_1 = f"""SELECT datetime, flow as '{gage_id}'
#             FROM read_parquet('{s3_path}', hive_partitioning=true)
#             WHERE gage='{gage_id}' and event='{event_id}';"""
# df = conn.execute(query_1).fetchdf()


query_2 = f"""SELECT *
            FROM read_parquet('{s3_path}', hive_partitioning=true)
            WHERE event='{event_id}';"""

df = conn.execute(query_2).fetchdf()
pivot_df = df.pivot(index="datetime", columns="gage", values="flow")

conn.close()
