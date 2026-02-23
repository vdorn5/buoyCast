# s3_utils.py

from io import BytesIO
import boto3
from botocore.client import ClientError
import pandas as pd
import sagemaker


def get_s3_client(region_name: str = None) -> boto3.client:
    """
    Create an S3 client for the given region.

    Args:
        region_name (str, optional): AWS region name. Uses default if None.

    Returns:
        boto3.client: S3 client instance.
    """
    session = boto3.session.Session()
    if region_name is None:
        region_name = session.region_name

    s3_client = boto3.client("s3", region_name=region_name)
    return s3_client


def get_sagemaker_bucket(sagemaker_session: sagemaker.Session = None) -> str:
    """
    Retrieve the default SageMaker bucket. Creates one if it doesn't exist.

    Args:
        sagemaker_session (sagemaker.Session, optional): SageMaker session. Creates default if None.

    Returns:
        str: S3 bucket name.
    """
    if sagemaker_session is None:
        sagemaker_session = sagemaker.Session()

    bucket = sagemaker_session.default_bucket()
    return bucket


def verify_bucket(bucket: str, s3_client: boto3.client = None) -> bool:
    """
    Verify that an S3 bucket exists.

    Args:
        bucket (str): Name of the bucket.
        s3_client (boto3.client, optional): S3 client instance. Creates one if None.

    Returns:
        bool: True if bucket exists, False otherwise.
    """
    if s3_client is None:
        s3_client = get_s3_client()

    try:
        s3_client.head_bucket(Bucket=bucket)
        print(f"[INFO] Bucket '{bucket}' exists.")
        return True
    except ClientError as e:
        print(f"[ERROR] Cannot find bucket '{bucket}': {e}")
        return False


def upload_df_to_s3(
    df: pd.DataFrame,
    bucket: str,
    s3_prefix: str,
    buoy_id: str,
    file_name: str = "stdmet.parquet",
    file_format: str | None = None,
    s3_client: boto3.client = None,
) -> str:
    """
    Upload a DataFrame to S3 as either Parquet or CSV.

    The object key is created using a Hive-style partition folder:

        {s3_prefix}/buoy={buoy_id}/{file_name}

    Parameters
    ----------
    df:
        DataFrame to upload.
    bucket:
        S3 bucket name.
    s3_prefix:
        Prefix/path in the bucket.
    buoy_id:
        Buoy identifier for folder structure.
    file_name:
        File name to write (e.g., "stdmet.parquet" or "stdmet.csv").
    file_format:
        Either "parquet" or "csv". If None, inferred from file_name extension.
    s3_client:
        Optional boto3 S3 client. If None, one is created.

    Returns
    -------
    str
        Full S3 URI of the uploaded object.
    """
    if s3_client is None:
        s3_client = get_s3_client()

    if file_format is None:
        if file_name.lower().endswith(".parquet"):
            file_format = "parquet"
        elif file_name.lower().endswith(".csv"):
            file_format = "csv"
        else:
            raise ValueError(
                "Could not infer file_format from file_name. "
                "Provide file_format='parquet' or 'csv'."
            )

    file_format = file_format.lower().strip()
    buffer = BytesIO()

    if file_format == "parquet":
        # Requires a parquet engine (pyarrow or fastparquet) installed.
        df.to_parquet(buffer, index=False)
    elif file_format == "csv":
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        buffer.write(csv_bytes)
    else:
        raise ValueError(f"Unsupported file_format='{file_format}'. Use 'parquet' or 'csv'.")

    buffer.seek(0)

    s3_key = f"{s3_prefix}/buoy={buoy_id}/{file_name}"

    s3_client.put_object(Bucket=bucket, Key=s3_key, Body=buffer.getvalue())

    s3_uri = f"s3://{bucket}/{s3_key}"
    print(f"[INFO] Uploaded to {s3_uri}")
    return s3_uri
