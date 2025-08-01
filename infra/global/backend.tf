terraform {
  backend "s3" {
    bucket         = "mlops-global-tfstate"           # S3 bucket to store terraform.tfstate
    key            = "fraud_detection/terraform.tfstate"  # Folder-like path within the bucket
    region         = "us-east-1"                      # S3 bucket region
    dynamodb_table = "mlops-global-lock"              # DynamoDB table to lock state
  }
}
