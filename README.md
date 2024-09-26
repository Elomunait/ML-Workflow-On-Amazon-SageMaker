# ML-Workflow-On-Amazon-SageMaker

This project demonstrates how to build a Machine Learning (ML) workflow on **Amazon SageMaker** to automate the process of training, deploying, and monitoring models for a fictional company called Scones Unlimited.

## Project Overview

The main goal of this project is to create an ML workflow for Scones Unlimited, an organization delivering scones using drone technology. The ML model predicts whether a drone's image contains a scone or not. The workflow built on **Amazon SageMaker** encompasses several stages, including data preprocessing, model training, and real-time inference using endpoints.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Architecture](#project-architecture)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Model Deployment](#model-deployment)
- [Model Monitoring](#model-monitoring)
- [Visualization](#visualization)
- [Results](#results)
- [References](#references)

## Project Overview

This project leverages **Amazon SageMaker** to build a robust ML pipeline for automating image classification tasks. The steps include:

1. Preprocessing the data.
2. Training an image classification model using SageMaker.
3. Deploying the trained model to a SageMaker endpoint.
4. Monitoring and capturing data from the deployed model.
5. Visualizing key metrics to evaluate the model performance.

## Project Architecture

The architecture of this project includes:

- **Amazon SageMaker Studio**: Used for creating and managing Jupyter notebooks for training and deployment.
- **SageMaker Model Monitor**: To track and evaluate model performance over time.
- **AWS Lambda**: Used to implement custom inference logic and integrate with Step Functions.
- **Step Functions**: For orchestrating the entire ML workflow, from inference to monitoring.
  
![Architecture](https://example-link-to-your-architecture-diagram.com)

## Getting Started

### 1. Clone the repository
To get started, clone this repository to your local machine using:

```bash
git clone https://github.com/yourusername/scones-unlimited-sagemaker-workflow.git
```

### 2. Install required packages

Before running the project, ensure you have the required Python dependencies installed. You can use the `requirements.txt` file to set up the environment:

```bash
pip install -r requirements.txt
```

### 3. Set up AWS Environment

Ensure that you have your AWS credentials configured and the following services enabled:

- SageMaker
- Step Functions
- AWS Lambda
- S3 Bucket (for data storage)

## Prerequisites

To replicate this project, ensure you have the following:

- An AWS account with permissions for Amazon SageMaker, Lambda, and Step Functions.
- Familiarity with Python (3.x) and Jupyter Notebooks.
- Boto3 and AWS CLI configured with credentials
- Installed libraries:
  - `boto3`
  - `sagemaker`
  - `tensorflow` or `PyTorch`

## Model Deployment

To deploy the ML model using Amazon SageMaker, follow these steps:

1. **Data Preprocessing**: Prepare your dataset and upload it to an S3 bucket.
2. **Model Training**: Use SageMaker to train the model. You can use the `train_model.ipynb` notebook to kickstart this process.
3. **Endpoint Deployment**: Deploy your trained model to an endpoint for real-time inference.

Here’s an example of how to deploy a model in SageMaker:

```python
deployment = img_classifier_model.deploy(
    instance_type="ml.m5.xlarge",
    initial_instance_count=1,
    endpoint_name="image-classification-endpoint",
    data_capture_config=data_capture_config
)

endpoint = deployment.endpoint_name
print("Endpoint Name:", endpoint)
```

## Model Monitoring

After deploying the model, **Model Monitor** will track the model’s input and output. This helps to ensure the model remains accurate over time.

- Configure the data capture settings for your endpoint.
- Set up SageMaker Model Monitor to evaluate data drift and anomalies.

## Visualization

Visualization is key to understanding the performance of the deployed model. We provide custom visualizations for monitoring:

- **Box Plot**: Displays the distribution of confidence levels for model predictions.
- **Histogram**: Shows the frequency of predictions that meet the defined confidence threshold.

```python
import plotly.express as px
box_fig = px.box(df, y="Confidence", points="all", title="Confidence Level Distribution")
box_fig.show()
```

## Results

- **Confidence Scores**: The model achieved confidence scores of as high as **99.8%** and as low as **69.6%** on various test images.
- **Visualization**: Monitoring visualizations indicate consistent performance with few outliers.

## References

This project was inspired by the **AWS SageMaker** documentation and relevant courses. Additional resources include:

- [Amazon SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [Boto3 - AWS SDK for Python](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [Step Functions Developer Guide](https://docs.aws.amazon.com/step-functions/)



## Troubleshooting

- **SageMaker Permissions**: Make sure your SageMaker role has the necessary permissions to access S3, Lambda, and other required services.
- **Endpoint Issues**: If you encounter issues with the endpoint, check the logs in CloudWatch for error details.

## References

- [Amazon SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [AWS Lambda](https://aws.amazon.com/lambda/)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Elomunait/ML-Workflow-On-Amazon-SageMaker/blob/main/LICENSE) file for details.
