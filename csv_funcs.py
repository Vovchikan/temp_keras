# system modules
import os
import re
import sys
from pathlib import Path

# anaconda modules
import pandas as pd

# Global variables
chemical_file = 'chemic.csv'
termal_file = 'termal.csv'
output_dir = './output'

def check_file (file_path: str) -> int:
  file = Path(file_path)
  if (not file.exists()):
    raise FileNotFoundError
  if (not file_path.endswith (".csv")):
    raise TypeError

  return 0

def save_df_to_csv (df: pd.DataFrame, output_f):
  completeName = os. path. join(output_dir, output_f)
  directory = os.path.dirname(completeName)
  if not os.path.exists(directory):
      os.makedirs(directory)
  df.to_csv (completeName, encoding='utf-8-sig')

def open_csv_file (csv_file):
  os.system ("start " + csv_file)

def delete_unused_collumns (df: pd.DataFrame) -> pd.DataFrame:
  normed_colls = filter (
    lambda name:
      re.search('' + 
      r'[E,e][N,n][D,d]\Z|' +
      r'[B,b][E,e][G,g]\Z|' +
      r'Ni|_n_|Ru|' +
      r'NUM', name) == None
    , df.columns)

  newdf = df[normed_colls]
  return newdf

def get_df_with_termal_elements (df: pd.DataFrame) -> pd.DataFrame:
  termal_colls = filter (
    lambda coll:
      re.search('' +
      r'(?!Name)' +
      r'[-d]', coll) != None
    , df.columns)
  
  newdf = df[termal_colls]
  return newdf
def get_df_with_chemical_elements (df: pd.DataFrame) -> pd.DataFrame:
  chemical_colls = filter (
    lambda coll:
      re.search('' +
      r'Name|' +
      r'[-d]', coll) == None
    , df.columns)

  newdf = df[chemical_colls]
  return newdf

# Delete rows based on percentage of NaN values in rows.
def delete_empty_rows (df: pd.DataFrame, perc: float) -> pd.DataFrame:
  rows = 0
  min_count =  int(((100-perc)/100)*df.shape[1] + 1)
  newdf = df.dropna (axis=rows,
                     thresh=min_count)
  return newdf

def create_training_df (df: pd.DataFrame) -> pd.DataFrame:
  chemical_df = get_df_with_chemical_elements (df)
  termal_df = get_df_with_termal_elements (df)

  try:
    save_df_to_csv (chemical_df, chemical_file)
    save_df_to_csv (termal_df, termal_file)
  except:
    print ("Error: ", sys.exc_info()[0])

  def delete_char_from_str (word:str, char) -> str:
    return word.replace(char, '')

  
  new_collumns = chemical_df.columns.append (pd.Index(['temp', 'sigma']))
  newdf = pd.DataFrame (columns=new_collumns)
  for i in range (len(chemical_df.index.values)):
    for j in range (len(termal_df.columns.values)):
      newdf.loc[str(i*j)] = chemical_df.iloc[i].tolist() + [int (delete_char_from_str (termal_df.columns[j], 'd')), termal_df.iloc[i,j] ]

  newdf = delete_empty_rows (newdf, 1)
  newdf = newdf.astype ({'temp':'int'})

  return newdf