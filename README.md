# SalesPrediction_wallmart
Sales Prediction PoC 


=== Description

This is a Docker Compose file that sets up an environment for training a sales prediction model with Python, Pandas, Dask, LightGBM, Optuna, and MLflow. The environment includes a Jupyter Notebook server, an MLflow tracking server, a MinIO object storage server, and a PostgreSQL database server.

=== Instructions

Install Docker and Docker Compose on your system if you haven't already done so.
Clone the repository containing the Docker Compose file.
Navigate to the directory containing the Docker Compose file.
Create a .env file in the same directory with the following contents:
makefile

Code:
>>POSTGRES_USER=your_postgres_username
>>POSTGRES_PASSWORD=your_postgres_password
>>POSTGRES_DB=your_postgres_database_name
>>MINIO_ACCESS_KEY=your_minio_access_key
>>MINIO_SECRET_KEY=your_minio_secret_key
>>Replace your_postgres_username, your_postgres_password, your_postgres_database_name, your_minio_access_key, and your_minio_secret_key with your own values.

Start the environment by running the following command:

Code:
>>docker-compose up -d

This command will start the Jupyter Notebook server, MLflow tracking server, MinIO object storage server, and PostgreSQL database server as Docker containers in the background. You will be able to access the Jupyter Notebook server at http://localhost:8888 and the MLflow tracking server at http://localhost:5000.

Open a web browser and navigate to http://localhost:8888. Enter the token displayed in the console when you started the Jupyter Notebook server.
Upload your sales data files to the data directory in the Jupyter Notebook server. Use the provided ETL script to preprocess the data and store it in the PostgreSQL database.
Use the provided MLflow experiment to train a sales prediction model with LightGBM and Optuna. The best model and its hyperparameters will be logged to the MLflow tracking server.