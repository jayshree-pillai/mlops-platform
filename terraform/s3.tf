resource "aws_s3_bucket" "mlops_fraud_dev" {
  bucket = "mlops-fraud-dev"
  force_destroy = true

  tags = {
    Name        = "mlops-fraud-dev"
    Environment = "dev"
  }
}

resource "aws_s3_bucket_versioning" "versioning" {
  bucket = aws_s3_bucket.mlops_fraud_dev.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_acl" "bucket_acl" {
  bucket = aws_s3_bucket.mlops_fraud_dev.id
  acl    = "private"
}

resource "aws_s3_bucket_public_access_block" "block_public" {
  bucket = aws_s3_bucket.mlops_fraud_dev.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Optional: Predefined folder structure via dummy objects
resource "aws_s3_object" "folders" {
  for_each = toset([
    "data/raw/",
    "data/processed/",
    "models/",
    "monitoring/",
    "drift_reports/",
    "explanations/"
  ])

  bucket = aws_s3_bucket.mlops_fraud_dev.id
  key    = each.key
  content = ""
}
