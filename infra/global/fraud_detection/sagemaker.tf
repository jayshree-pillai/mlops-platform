resource "aws_iam_role" "sagemaker_exec_role" {
  name = "sagemaker-fraud-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_s3_access" {
  role       = aws_iam_role.sagemaker_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_sagemaker_model" "fraud_model" {
  name              = "fraud-byoc-model"
  execution_role_arn = aws_iam_role.sagemaker_exec_role.arn

  primary_container {
    image          = "<your-ecr-image-uri>"  # Replace after pushing
    mode           = "SingleModel"
    model_data_url = "s3://mlops-fraud-dev/mlruns/0/123abc456def789/artifacts/model/"  # real MLflow path

    environment = {
      MODEL_BUCKET  = "mlops-fraud-dev"
      MODEL_KEY     = "mlruns/0/123abc456def789/artifacts/model/model.pkl"
      SCHEMA_KEY    = "mlruns/0/123abc456def789/artifacts/model/features.pkl"
      MODEL_VERSION = "v1"
    }
  }
}

resource "aws_sagemaker_endpoint_configuration" "fraud_config" {
  name = "fraud-config"
  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.fraud_model.name
    initial_instance_count = 1
    instance_type          = "ml.m5.large"
  }
}

resource "aws_sagemaker_endpoint" "fraud_endpoint" {
  name = "fraud-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.fraud_config.name
}
