# Braket Backend Configuration

The `BraketBackend` allows you to execute HLQuantum circuits using Amazon Braket, both on local simulators and quantum hardware on AWS.

## Installation

Ensure you have the required extra installed:
```bash
pip install hlquantum[braket]
```
This installs the `amazon-braket-sdk`.

## Local Simulator

By default, without any configuration, the backend runs using the local Braket simulator.

```python
from hlquantum.backends import BraketBackend
import hlquantum as hlq

# Uses Braket's LocalSimulator
backend = BraketBackend()
result = hlq.run(circuit, backend=backend)
```

## AWS Quantum Hardware (QPU) and Cloud Simulators

To execute on real AWS quantum hardware or AWS-managed simulators (e.g., SV1, TN1), you must configure your AWS credentials.

### AWS Credentials

You need an active AWS account with the Amazon Braket service enabled. Ensure your environment is configured with your AWS access keys:

1. **Access Key ID** (`AWS_ACCESS_KEY_ID`)
2. **Secret Access Key** (`AWS_SECRET_ACCESS_KEY`)
3. **AWS Region** (`AWS_DEFAULT_REGION`)

You can set these via the AWS CLI (`aws configure`) or environment variables:
```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="us-east-1"
```

### Running on a Device

Instantiate an `AwsDevice` using its ARN and pass it to the `BraketBackend`. You should also provide an S3 bucket destination to store the results.

```python
from braket.aws import AwsDevice
from hlquantum.backends import BraketBackend
import hlquantum as hlq

# Instantiate an IonQ device on AWS Braket
ionq_device = AwsDevice("arn:aws:braket:::device/qpu/ionq/Harmony")

# S3 bucket for storing task results: (bucket_name, prefix)
s3_folder = ("my-braket-bucket", "results-prefix")

backend = BraketBackend(device=ionq_device, s3_destination=s3_folder)
result = hlq.run(circuit, backend=backend)
```
