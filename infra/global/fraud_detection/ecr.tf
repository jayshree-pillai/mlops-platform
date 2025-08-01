resource "aws_ecr_repository" "fraud_repo" {
  name = "fraud-detection"
}

resource "aws_ecr_lifecycle_policy" "fraud_policy" {
  repository = aws_ecr_repository.fraud_repo.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Expire untagged images after 30 days"
        selection = {
          tagStatus     = "untagged"
          countType     = "sinceImagePushed"
          countUnit     = "days"
          countNumber   = 30
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}
