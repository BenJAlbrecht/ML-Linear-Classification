# Importing necessary libraries.
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

""" Classes are listed in order used.

classes:
~~~~~~~~~~~~~~

- data_class
--- Generates and visualizes dataset.

- wtd_model
--- Model class.

"""


# Data creation and visualization class.
class data_class():

  def __init__(self, points = 100, dimension = 2, sep = 0.05):
    # Dataframes.
    self.temp_df = None           # Temporary storage dataframe.
    self.df = None                # Final data dataframe.

    # Settings.                 
    self.points = points          # Number of points to generate.
    self.dimension = dimension    # Dimensions to generate within.
    self.sep = sep                # Quantity to ensure linear separability.


  # Function to generate data.
  def generate_data(self):
    pre_df = np.random.rand(self.points, self.dimension)
    self.temp_df = pd.DataFrame(pre_df)

    line = self.draw_line()
    self.temp_df = self.classify(line, self.temp_df)

    self.adjust()
    box_mask = (self.temp_df.iloc[:,0]<=1)&(self.temp_df.iloc[:,0]>=0)&(self.temp_df.iloc[:,1]<=1)&(self.temp_df.iloc[:,1]>=0)
    self.temp_df = self.temp_df[box_mask]
    self.temp_df.reset_index(drop=True, inplace=True)

    v = self.valid_check(self.temp_df)
    if(0.60*self.points >= v >= 0.40*self.points):
      self.df = self.temp_df
      return
    else:
      self.generate_data()


  # Generate hyperplane which establishes the "ground truth" of the data.
  def draw_line(self):
    return np.random.uniform(-1, 1, self.dimension)


  # Classifies data according to relation to hyperplane.
  def classify(self, line, df):
    df['y'] = np.where((line[0]*df[0]+line[1]<df[1]), 1, 0)
    return df


  # Helper with checking if we have roughly even data.
  def valid_check(self, df):
    return df['y'].sum()


  # Adjust our data to ensure linear separability.
  def adjust(self):
    mask = self.temp_df['y'] == 1
    self.temp_df.loc[mask, 1] += self.sep


  # Graph dataset.
  def visualize(self):
    colors = ['red', 'blue']
    plt.scatter(self.df[0], self.df[1], c=self.df['y'].apply(lambda x: colors[x]), s=12)
    plt.title('Random Points')
    plt.show()



# Model class.
class wtd_model:

  def __init__(self, data, iter, k = 100):

    # Initial settings.
    self.data = data        # Dataframe passed to model.
    self.iter = iter        # Number of iterations to complete.
    self.k = k              # Number of cuts per iteration.

    # Results.
    self.m = None           # Final slope.
    self.b = None           # Final intercept.
        
    # Helper variable.
    self.feas_cuts = None   # Cuts that are "feasible".


  # main fitting function
  def fit(self):
    n = self.iter
    cols = ['m', 'b', 'e']
    cut_data = np.zeros((n, len(cols)))
    cuts = pd.DataFrame(cut_data, columns=cols)
    for idx, row in cuts.iterrows():
      (M, B) = self.cut()
      M = self.check_slope(M,B)
      cuts.at[idx, 'm'] = M
      cuts.at[idx, 'b'] = B
      y_ = self.pred(M, B)
      cuts.at[idx, 'e'] = self.error(y_)
           
      self.feas_cuts = cuts
      best = self.feas_cuts['e'].idxmin()
      self.best_cut(best)


    # Define best cut.
  def best_cut(self, best):
    self.m = self.feas_cuts.at[best, 'm']
    self.b = self.feas_cuts.at[best, 'b']
           

  # Create cuts.
  def cut(self):
    l,u,n = -1,1,self.k
    m,b = np.random.uniform(l,u,n),np.random.uniform(l,u,n)

    df = pd.DataFrame({'m': m, 'b': b})
    df['v'] = df.apply(lambda x: self.valid(x['m'], x['b']), axis = 1)
    df['v'] = df['v'].astype(int)
    valid = df.loc[df['v']!=0].reset_index(drop=True)

    v2 = self.make_wts(valid)
    v3 = self.wt_line(v2)

    sum_w = v3['w'].sum()
    avg_m = (v3['wt_m'].sum()) / sum_w
    avg_b = (v3['wt_b'].sum()) / sum_w

    return avg_m, avg_b


  # USe errors to weight lines.
  def wt_line(self, df):
    df['wt_m'] = 0
    df['wt_b'] = 0
    for idx, row in df.iterrows():
      m, b, w = row['m'], row['b'], row['w']
      if w != 0:
        wtd_m, wtd_b = w*m, w*b
        df.at[idx, 'wt_m'] = wtd_m
        df.at[idx, 'wt_b'] = wtd_b
      else:
        df.at[idx, 'wt_m'] = m
        df.at[idx, 'wt_b'] = b
    return df
    

  # Generate weights for weighted average.
  def make_wts(self, df):
    df['w'] = 0
    for idx, row in df.iterrows():
      m_val = row['m']
      b_val = row['b']
      y_ = self.pred(m_val, b_val)
      E_ = self.error(y_)
      if E_ != 0:
        df.at[idx, 'w'] = 1 / E_
      else:
        df.at[idx, 'w'] = 0
    return df
    

  # Check if cuts are valid within box.
  def valid(self, m, b):
    if (0 <= b <= 1) or (0 <= m+b <= 1):
      return True
    if (0 <= -b/m <= 1) or (0 <= (1-b)/m <= 1):
      return True
    return False
    
    
  # Prediction function.
  def pred(self, m ,b):
    return np.where((m*self.data[0]+b<self.data[1]), 1, 0)
    

  # Error function.
  def error(self, y_):
    return (sum(self.data['y'] != y_))/len(self.data['y'])
    

  # Check slope.
  def check_slope(self, m, b):
    test_m = -1 * m

    y_true = self.pred(m,b)
    y_test = self.pred(test_m,b)

    E_true = self.error(y_true)
    E_test = self.error(y_test)

    if E_true <= E_test:
      return m
    if E_true > E_test:
      return test_m
    
  
  # Testing prediction and error functions.
  def test_pred(self, test):
    return np.where((self.m*test[0] + self.b < test[1]), 1, 0)
  def test_error(self, y_pred, y_true):
    testing_error = sum(y_pred != y_true) / len(y_true)
    return testing_error