#     In this Directory, the linear regression algorithm is employed to predict which labels would have the most
# influence on a player's Hours Spent Per Week on a videogame.  The dataset is a csv containing data from the
# Skillcraft 2 installment.  Further, the csv was obtained from the UC Irvine repo, and it turns out that age didn't
# have as much influence on HPW as I hypothesized.
#     For configuration, one first needs to install Anaconda: this distribution will contain not only Python, but
# all the libraries relevant to machine learning.  Other components that will need to be installed at some point are:
# Tensor Flow
# Keras
# Pip
# Note that Python 3.5-3.7 are, supposedly, the only versions that work correctly with TF, but the Anaconda download
# should account for that.
#     In short, the pandas library reads the csv file and the training methods from the sklearn library will produce
# linear regression model based on the chosen labels.  From there its accuracy can be printed to the console and/or
# the weights of coefficients that effect the outcome.  Matplotlib can also be used to bring up a visual graph of the
# data.