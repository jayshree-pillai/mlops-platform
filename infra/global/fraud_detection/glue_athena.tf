resource "aws_glue_catalog_database" "fraud_db" {
  name = "fraud_features_db"
}

resource "aws_glue_catalog_table" "fraud_fg_table" {
  name          = "fraud_features"
  database_name = aws_glue_catalog_database.fraud_db.name

  table_type = "EXTERNAL_TABLE"

    storage_descriptor {
      location      = "s3://mlops-fraud-dev/feature_store/"
      input_format  = "org.apache.hadoop.mapred.TextInputFormat"
      output_format = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"

     column {
          name = "step"
          type = "int"
        }
        column {
          name = "type"
          type = "string"
        }
        column {
          name = "amount"
          type = "double"
        }
        column {
          name = "nameOrig"
          type = "string"
        }
        column {
          name = "oldbalanceOrg"
          type = "double"
        }
        column {
          name = "newbalanceOrig"
          type = "double"
        }
        column {
          name = "nameDest"
          type = "string"
        }
        column {
          name = "oldbalanceDest"
          type = "double"
        }
        column {
          name = "newbalanceDest"
          type = "double"
        }
        column {
          name = "isFraud"
          type = "int"
        }
        column {
          name = "isFlaggedFraud"
          type = "int"
        }
        column {
          name = "tx_id"
          type = "string"
        }
        column {
          name = "timestamp"
          type = "string"
        }
        column {
          name = "dataset_split"
          type = "string"
        }

      ser_de_info {
        serialization_library = "org.openx.data.jsonserde.JsonSerDe"
      }
    }

}
    resource "aws_athena_workgroup" "fraud_athena" {
      name = "fraud_athena"
      configuration {
        result_configuration {
          output_location = "s3://mlops-fraud-dev/athena_results/"
        }
      }
    }

#IF U need to add additonal columns:
#aws athena start-query-execution \
#  --query-string "ALTER TABLE fraud_features ADD COLUMNS (tx_id string, timestamp string, dataset_split string);" \
#  --query-execution-context Database=fraud_features_db \
#  --result-configuration OutputLocation=s3://mlops-fraud-dev/athena_results/
