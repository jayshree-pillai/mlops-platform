resource "aws_ecr_repository" "fraud_repo" {
  name                 = "fraud-byoc"
  image_tag_mutability = "MUTABLE"

  lifecycle_policy {
    policy = jsonencode({
      rules = [
        {
          rulePriority = 1
          description  = "Expire untagged images after 7 days"
          selection    = {
            tagStatus = "untagged"
            countType = "sinceImagePushed"
            countUnit = "days"
            countNumber = 7
          }
          action = {
            type = "expire"
          }
        }
      ]
    })
  }
}
