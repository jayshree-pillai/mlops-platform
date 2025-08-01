resource "aws_cloudwatch_event_rule" "daily_fraud_retrain" {
  name                = "daily-fraud-retrain"
  description         = "Trigger fraud retrain Step Function every 24 hours"
  schedule_expression = "rate(1 day)"
}

resource "aws_cloudwatch_event_target" "trigger_retrain_stepfn" {
  rule      = aws_cloudwatch_event_rule.daily_fraud_retrain.name
  arn       = aws_sfn_state_machine.fraud_retrain_sm.arn
  role_arn  = aws_iam_role.stepfn_exec_role.arn
}

resource "aws_iam_role_policy_attachment" "stepfn_event_trigger" {
  role       = aws_iam_role.stepfn_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/AWSStepFunctionsFullAccess"
}
