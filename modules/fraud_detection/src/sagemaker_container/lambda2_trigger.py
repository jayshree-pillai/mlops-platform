import boto3, json

sm = boto3.client("sagemaker-runtime")
response = sm.invoke_endpoint(
    EndpointName="fraud-byoc-endpoint",
    ContentType="application/json",
    Body=json.dumps({"features": [1.0, 2.0, ...]})
)
result = json.loads(response["Body"].read())
