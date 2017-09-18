# AWS Machine Learning / Google Cloud ML Engine demo

This repository contains the data & code needed for the Alma Talent workshop on AWS Machine Learning and Google Cloud Machine Learning engine. The general idea is to use the classic Iris dataset to train a classifier using both systems, and to deploy both to the cloud for an instant, fully managed inference API. To do the demo yourself, you need credentials for Google Cloud and AWS Machine Learning.

    gcloud ml-engine versions create v1 --model=alma_demo_model --origin=model --staging-bucket=gs://alma-demo --verbosity=debug

    gcloud ml-engine predict --model=alma_demo_model --version=v1 --json-instances=test-data.json