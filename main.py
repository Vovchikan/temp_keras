# system modules
import argparse

# anaconda modules
import pandas as pd

# user modules
import csv_funcs
import train

# Global variables
default_input_file = 'data.csv'
output_file = 'temp.csv'
corr_file = 'corr.csv'
perc_of_nan = 35

def main (show, input_file):

  try:
    csv_funcs.check_file (input_file)
  except TypeError:
    print ('Wrong extension - ', input_file)
    print ('Extension must be ".csv"')
    exit ()
  except FileNotFoundError:
    print ('File - ', input_file, ' doesn\'t exist!')
    exit ()

  df = pd.read_csv (input_file, encoding='utf-8')

  mod_df = csv_funcs.delete_unused_collumns (df)
  mod_df = csv_funcs.delete_empty_rows (mod_df, perc_of_nan)
  train_df = csv_funcs.create_training_df (mod_df)

  train.start (train_df, show)
  
  cor_df = train_df.corr()
  try:
    csv_funcs.save_df_to_csv (train_df, output_file)
    csv_funcs.save_df_to_csv (cor_df, corr_file)
  finally:
    print ('End.')
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser ()
  parser.add_argument('--show', required=False,
                      help='show graphics', default=False,
                      action='store_true')
  parser.add_argument('--db', metavar='path', required=False,
                      help='path to file with data', type=str, default=default_input_file)
  args = parser.parse_args()
  main (show=args.show, input_file=args.db)