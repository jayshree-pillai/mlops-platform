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

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_sagemaker_model" "byoc_model" {
  name               = "fraud-byoc-model"
  execution_role_arn = aws_iam_role.sagemaker_exec_role.arn

  primary_container {
    image          = "869935087425.dkr.ecr.us-east-1.amazonaws.com/fraud-byoc:latest"
    mode           = "SingleModel"
    model_data_url = null  # not needed for BYOC container

    environment = {
      MODEL_BUCKET     = "mlops-fraud-dev"
      MODEL_KEY        = "mlruns/0/123abc456def789/artifacts/model/model.pkl"
      SCHEMA_KEY       = "mlruns/0/123abc456def789/artifacts/model/features.pkl"
      MODEL_VERSION    = "v1"
      PUSHGATEWAY_URL  = "http://52.22.35.87:9091"  # This is PushGateway, not Prometheus
    }
  }
}

resource "aws_sagemaker_endpoint_configuration" "byoc_config" {
  name = "fraud-byoc-config"

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.byoc_model.name
    initial_instance_count = 1
    instance_type          = "ml.m5.large"
  }
}

resource "aws_sagemaker_endpoint" "byoc_endpoint" {
  name                 = "fraud-byoc-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.byoc_config.name
}
