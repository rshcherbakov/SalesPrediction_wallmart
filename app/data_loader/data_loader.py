import pandas as pd
from loguru import logger


class SalesDataLoader:
    """
       Basic example of dataloader
       
       Note 1 index, datetime fields, 
       columns and data tipes should be specified as well

       Note 2 if quantity of datasources 2 or more it's 
       will be a good idea make a basic abstract class for them to make sure that the interface is standard 
    """
    def __init__(self, filename):
        self.filename = filename
        self.data = None  # as an option _data could be a privat +
                          # @property methods could be setted 
    
    def load_data(self):
        """_basic dataloader

        Returns:
            pd.DataFrame: loaded data in pandas dataset format 
        """
        sales_df = pd.read_csv(self.filename)  
        self.data = sales_df 
        logger.debug("data successfully loaded")
        return sales_df
   