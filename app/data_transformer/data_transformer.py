import pandas as pd
from loguru import logger

class SalesDataTransformer:
    """Set of methods for data transform
       Could be specific for datasources

    """
    def __init__(self):
        pass
    
    def transform_data(self, sales_df):
        """Data transformation method

        Args:
            sales_df (pd.DataFRame): dataset that should be transformed

        Returns:
            pd.DataFrame: transformed dataset
        """
        sales_df['date'] = pd.to_datetime(sales_df['date'])
        sales_df['revenue'] = sales_df['units_sold'] * sales_df['price']
        features_df = sales_df[['name', 'category', 'store', 'date', 'revenue']]
        monthly_sales = features_df.groupby(['name', 'category', 'store', pd.Grouper(key='date', freq='M')]).sum()
        
        # !Note: probably could be helpfull to add time/order lag features 
        # !Note: index should be set to date/order and at least item/store
        logger.debug("Dataset successfully transformed")
        return monthly_sales