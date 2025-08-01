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
