resource "aws_db_instance" "mlflow_postgres" {
  identifier              = "mlflow-db"
  engine                  = "postgres"
  instance_class          = "db.t3.micro"
  allocated_storage       = 20
  name                    = "mlflow_db"
  username                = "mlflow_user"
  password                = "mlflow_pass123"
  skip_final_snapshot     = true
  publicly_accessible     = true
  vpc_security_group_ids  = [aws_security_group.mlflow_db_sg.id]
  deletion_protection     = false
  backup_retention_period = 0
}

resource "aws_security_group" "mlflow_db_sg" {
  name        = "mlflow-db-sg"
  description = "Allow EC2 to access Postgres"
  vpc_id = "vpc-0fe2c0d64434886e1"


  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["3.215.110.164/32"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
