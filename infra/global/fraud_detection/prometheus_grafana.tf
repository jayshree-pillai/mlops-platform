resource "aws_security_group" "monitoring_sg" {
  name        = "prometheus-grafana-sg"
  description = "Allow access to Prometheus and Grafana"
  vpc_id      = "vpc-0fe2c0d64434886e1"

  ingress {
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = ["67.81.86.5/32"]
  }

  ingress {
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = ["67.81.86.5/32"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "prometheus-grafana"
  }
}
resource "aws_instance" "monitoring_ec2" {
  ami                    = "ami-020cba7c55df1f615"
  instance_type          = "t3.micro"
  subnet_id              = "subnet-0d4cee39fd0c06d57"
  vpc_security_group_ids = [aws_security_group.monitoring_sg.id]
  key_name               = "jp-ec2-key"

  tags = {
    Name = "prometheus-grafana"
    Role = "monitoring"
  }
}
