import pandas as pd
from collections import List
from dataclasses import dataclass

@dataclass 
class DF_Grupper:
    df:pd.DataFrame # Basic pd.DataFrame
    group_cols:List[str] # columns for grupping
    date_cols:List[str] # columns with date  
  
    def lag_leatures_creator(self, 
                             n_lags:int=3, 
                             lag_cols:List[str]=None) -> pd.DataFrame:
        """method for making lag features 
        Args:
            n_lags (int, optional): number of time periods. Defaults to 3.
            lag_cols (List[str], optional): names of columns for lag variables. Defaults to None.

        Returns:
            pd.DataFrame: new dataframe 
        """

        df_c = self.df.copy(deep=True)
        grouped_dfs = self.df.groupby(self.group_cols)

        for col in lag_cols: 
            for i in range(1,n_lags):
                df_c[f'lag{i}_{col}'] = grouped_dfs[col].shift(i)
        return df_c
        
