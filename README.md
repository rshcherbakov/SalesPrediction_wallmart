# SalesPrediction_wallmart
Sales Prediction PoC 

> **Note:** **TBD**: actual data transformation and 
loading methods will be ajusted for 
[dataset](https://www.kaggle.com/datasets/divyajeetthakur/walmart-sales-prediction dataset)
However The template could be used for other perposes


> **TODO:**
> 1. Dataloader not adjusted for current data scheme
>    1. Add date fields while reading data from datasource
>    2. Read only needed columns
>    3. Set right columns data type while reading data  
>    4. Add index to the main datasource (date + item/store/group + timeperiod)
>    5. Sort by date 
> 2. Dataprocessors (Transformers) should be ajusted as well  
>    Order of transformation:
>    1. merge datasources
>    2. clean data/fill nan values/fix issues  
>    3. split train/val/test by slices (Important - do this before step 2.4)
>    4. add aggregation transformation and timelag features for each datafold
> 3. Model & Postprocessing
>    1. conduct several experiments
>    2. try stats model stecking with adding some other ML algorithm like SVR or MLP
>    3. Add feature importance and feature selection 
>    4. Add some metrics for business like **MAPE**, or/and some Business metrics
> 4. Add dashboard
>   1. Streamlit || Gradio || Graphana 
> 5. Add incomming data monitoring
> 6. Add Feedback loop and Trining loop automation   


## Repository Description

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
