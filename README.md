# SalesPrediction_wallmart
Sales Prediction PoC 

<div class="alert alert-block alert-danger">
<b>Attention:</b> application container works in the dev mode (via tty) in order to simplify further development process. main.py entrypoint is temporary detached
</div>

> **Note:** **TBD**: actual data transformation and 
loading methods will be ajusted for 
https://www.kaggle.com/datasets/divyajeetthakur/walmart-sales-prediction
dataset
However The template could be used for other perposes


## Description

This is a Docker Compose file that sets up an environment for training a sales prediction model with Python, Pandas, LightGBM, Optuna, and MLflow. The environment includes a Jupyter Notebook server, an MLflow tracking server, a MinIO object storage server, and a PostgreSQL database server.

## Instructions

Install Docker and Docker Compose on your system if you haven't already done so.
Clone the repository containing the Docker Compose file.
Navigate to the directory containing the Docker Compose file.
Create a .env file in the same directory with the following contents:
makefile

Code:
>POSTGRES_USER=your_postgres_username
>POSTGRES_PASSWORD=your_postgres_password
>POSTGRES_DB=your_postgres_database_name
>MINIO_ACCESS_KEY=your_minio_access_key
>MINIO_SECRET_KEY=your_minio_secret_key

Replace your_postgres_username, your_postgres_password, your_postgres_database_name, your_minio_access_key, and your_minio_secret_key with your own values.

Start the environment by running the following command:

Code:
>docker-compose up -d


