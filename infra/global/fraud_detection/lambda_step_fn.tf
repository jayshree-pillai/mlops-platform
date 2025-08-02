resource "aws_iam_role" "stepfn_exec_role" {
  name = "stepfn_exec_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Allow",
      Principal = {
        Service = "states.amazonaws.com"
      },
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "stepfn_basic" {
  role       = aws_iam_role.stepfn_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSStepFunctionsFullAccess"
}

# simple, clean retraining pipeline
resource "aws_sfn_state_machine" "fraud_retrain_sm" {
  name     = "fraud-retrain-stepfn"
  role_arn = aws_iam_role.stepfn_exec_role.arn
  definition = file("${path.module}/step_function_definition.json")
}

resource "aws_cloudwatch_event_rule" "daily_retrain" {
  name                = "daily-fraud-retrain"
  schedule_expression = "rate(1 day)"
}

resource "aws_cloudwatch_event_target" "start_stepfn" {
  rule      = aws_cloudwatch_event_rule.daily_retrain.name
  arn       = aws_sfn_state_machine.fraud_retrain_sm.arn
  role_arn  = aws_iam_role.stepfn_exec_role.arn
}
