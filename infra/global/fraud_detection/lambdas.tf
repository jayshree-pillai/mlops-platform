resource "aws_iam_role" "lambda_exec" {
  name = "lambda_exec_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_lambda_function" "load_training_data" {
  function_name = "load-training-data"
  role          = aws_iam_role.lambda_exec.arn
  handler       = "main.lambda_handler"
  runtime       = "python3.10"
  filename      = "load_training_data.zip"
  source_code_hash = filebase64sha256("load_training_data.zip")
}

resource "aws_lambda_function" "preprocess_data" {
  function_name = "preprocess-data"
  role          = aws_iam_role.lambda_exec.arn
  handler       = "main.lambda_handler"
  runtime       = "python3.10"
  filename      = "preprocess_data.zip"
  source_code_hash = filebase64sha256("preprocess_data.zip")
}

resource "aws_lambda_function" "train_model" {
  function_name = "train-model"
  role          = aws_iam_role.lambda_exec.arn
  handler       = "main.lambda_handler"
  runtime       = "python3.10"
  filename      = "train_model.zip"
  source_code_hash = filebase64sha256("train_model.zip")
}

resource "aws_lambda_function" "evaluate_model" {
  function_name = "evaluate-model"
  role          = aws_iam_role.lambda_exec.arn
  handler       = "main.lambda_handler"
  runtime       = "python3.10"
  filename      = "evaluate_model.zip"
  source_code_hash = filebase64sha256("evaluate_model.zip")
}

resource "aws_lambda_function" "log_to_mlflow" {
  function_name = "log-to-mlflow"
  role          = aws_iam_role.lambda_exec.arn
  handler       = "main.lambda_handler"
  runtime       = "python3.10"
  filename      = "log_to_mlflow.zip"
  source_code_hash = filebase64sha256("log_to_mlflow.zip")
}
