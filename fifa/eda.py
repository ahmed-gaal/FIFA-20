import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import ppscore as pps
import sklearn
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LinearRegression, RANSACRegressor, Lasso, ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
plt.style.use('dark_background')

df = pd.read_csv('fifa.csv', sep = ';')
new_df = df.sort_values(by = 'overall', ascending = False)

class Eda():
    def __init__():
        '''Initializing the class'''
    
    def nations(country):
        '''
            Function to return countries that most players represent in fifa
                Args:
                    pandas.Series or DataFrame column
                ___________
                Returns:
                    pandas.DataFrame object
        '''
        nat = df[country].value_counts().head(10)
        nat = nat.to_frame(name = 'Total Players')
        return nat

    def popular_clubs(x):
        '''
            Function to return popular clubs in fifa
                Args:
                    pandas.Series or DataFrame column
                ___________
                Returns:
                    pandas.DataFrame object
        '''
        clb = df[x].value_counts().head(40)
        clb = clb.to_frame(name = 'Total Players')
        return clb

    def young_player(name):
        '''
            Function to return youngest players with highest potential
                ________________
                Returns:
                    pandas.DataFrame
        '''
        return df.nsmallest(10, columns = name).sort_values(by = 'potential', ascending = False)

    def old_squad(x,y):
        '''
            Function to return the team with the average oldest squad in fifa
                ________________ 
                Returns:
                    pandas.DataFrame
        '''
        ols = df.groupby([x])[y].mean().sort_values(ascending = False).head(10)
        ols = ols.to_frame(name = 'Age')
        return ols

    def best_cdm(pos, x):
        '''
            Function to return the best central defensive midfielders
                _______________ 
                Returns:
                    pandas.DataFrame
        '''
        return df[df[pos] == x][['name','age','team','overall','potential']].head(10)

    def best_str(pos, x):
        '''
            Function to return the best strikers
                ____________
                Returns:
                    pandas.DataFrame
        '''
        return df[df[pos] == x][['name','age','team','overall','potential']].head(10)

    def best_lw(pos, x):
        '''
            Function to return best left-wing forwards
                __________
                Returns:
                    pandas.DataFrame
        '''
        return new_df[new_df[pos] == x][['name','age','team','overall','potential']].head(10)

    def best_rw(pos, x):
        '''
            Function to return best right-wing forwards\
                _________
                Returns:
                    pandas.DataFrame
        '''
        return new_df[new_df[pos] == x][['name','age','team','overall','potential']].head(10)

    def best_cf(pos, x):
        '''
            Function to return best center forwards
                ___________
                Returns:
                    pandas.DataFrame
        '''
        return df[df[pos] == x][['name','age','team','overall','potential','age']].head(10)

    def best_cam(pos, x):
        '''
            Function to return best centre attacking midfielders
                __________
                Returns:
                    pandas.DataFrame
        '''
        return df[df[pos] == x][['name','age','team','overall','potential']].head(10)

    def country_squad(country):
        '''
            Function to return country team sheet
                Args:
                    pandas.Series or DataFrame column
                _____________
                Returns:
                    pandas.DataFrame
        '''
        return df.groupby('nationality').get_group(country).drop(['player_id'], axis = 1).head(23)

    def player_info(name):
        '''
            Function to return player information
                Args:
                    Player Name
                ____________
                Returns:
                    pandas.core.frame.DataFrame
        '''
        return df.loc[df['name']==name]

    def player_rate(x):
        '''
            Function for return player information based on overall ratings
                Args:
                    int: Player rating
                ____________
                Returns:
                    pandas.core.frame.DataFrame
        '''
        return df[df['overall'] == x][['name','overall','potential','position','team']].head(10)

    def player_age(x):
        '''
            Function for return player information based on age
                Args:
                    int: Player age
                ____________
                Returns:
                    pandas.core.frame.DataFrame
        '''
        return df[df['age'] == x][['name','nationality','team','position','age','overall']].head(10)

    def high_pot(x):
        '''
            Function to return players with highest potential
                Args:
                    pandas.Series or pandas.DataFrame column
                ____________
                Returns:
                    pandas.DataFrame
        '''
        return df.sort_values(x, ascending = False)[['name','age','team','position','overall','potential']].head(20)

    def pps_corr(x):
        '''
            Function for calculating the Predictive Power Score
                Args:
                    pandas.Series or DataFrame
                ___________
                Returns:
                    pandas.DataFrame or list of PPS dicts
        '''
        return pps.matrix(x) 

    def pps_heatmap(df):
        '''
            Function for calculating the Predictive Power Score and plotting a heatmap
                Args:
                    Pandas DataFrame or Series object
                __________
                Returns:
                    figure
        '''
        nuff = pps.matrix(df)
        nuff1 = nuff[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
        plt.figure(figsize = (15, 8))
        ax = sb.heatmap(nuff1, vmin=0, vmax=1, cmap="Oranges_r", linewidths=0.5, annot=True)
        ax.set_title("PPS matrix")
        ax.set_xlabel("feature")
        ax.set_ylabel("target")
        return ax

    def pairplot(x):
        '''
            Function for plotting a scatter matrix
                Args:
                    pandas.DataFrame
                ___________
                Returns:
                    figure
        '''
        plt.figure(figsize=(15,8))
        ax = sb.pairplot(x, height = 2.5)
        return ax

    def corr(x):
        '''
            Function for computing pairwise correlation of columns.
                Args:
                    pandas.DataFrame
                ____________
                Returns:
                    Correlation Matrix
        '''
        return x[['player_id','overall','name','team','age','hits','potential']].corr()

    def descriptive(x):
        '''
            Function for generating descriptive statistics.
                Args:
                    pandas.Series or pandas.DataFrame
                ________________
                Returns:
                    Summary statistics of the Series or Dataframe provided
        '''
        return x.describe()

    def heatmap(x):
        '''
            Function to return correlation matrix on a heatmap
                Args:
                    pandas.Series or DataFrame
                _____________
                Returns:
                    figure
        '''
        plt.figure(figsize = (12, 8))
        sb.heatmap(x.corr(), annot = True,cmap = 'Oranges')
        plt.show()

    def linear_regression(feature, target, z):
        '''
            Function for fitting a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the
            observed targets in the dataset, and the targets predicted by the linear approximation
        
                Args:
                    feature variable, target variable, prediction value(int)
                ________________
                Returns:
                    array: Prediction using the linear model
                    figure:
        '''
        ## Instantiating the algorithm ##
        model = LinearRegression()
        ## preparing the data and reshaping ##
        x = df[feature].values.reshape(-1, 1)
        y = df[target].values
        ## fitting the linear model ##
        model.fit(x, y)
        ## plotting the model ##
        plt.figure(figsize = (15, 8))
        sb.regplot(x, y)
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.title('Regression Plot')
        plt.show()
    
        return model.predict(np.array([z]).reshape(-1, 1))

    def regression_performance(feature, target, size, depth = None, model = str):
        '''
        Function to calculate the performance of a linear regression model.
            Args:
                feature variable, target variable, model type, test size, maximum depth of tree if model == tree
            _________________
            Returns:
                ❅ Mean squared error: (float)
                ❅ Coefficient of Determination R²: (float)
                ❅ Figure
        '''
        ## Arranging data and reshaping ##
        x = df[feature].values.reshape(-1, 1)
        y = df[target].values
        ## Splitting data into training and testing ##
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = size, random_state = 0)
        ## Instantiating the model ##
        if model == 'linear':
            model = LinearRegression()
        elif model == 'ransac':
            model = RANSACRegressor()
        elif model == 'tree':
            model = DecisionTreeRegressor(max_depth=depth)
        ## fitting the model ##
        model.fit(x_train, y_train)
        ## prediction results ##
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        ## Printing the MSE and the R² score ##
        print('Mean Squared Error: Training Data %.2f' %(mean_squared_error(y_train, y_train_pred)))
        print('Mean Squared Error: Test Data %.2f' %(mean_squared_error(y_test, y_test_pred)))
        print('Coefficient of Determination R²: Training Data %.2f' %(r2_score(y_train, y_train_pred)))
        print('Coefficient of Determination R²: Test Data %.2f' %(r2_score(y_test, y_test_pred)))
        ## Residual Analysis ##
        plt.figure(figsize = (10, 8))
        plt.scatter(y_train_pred, y_train_pred - y_train, c = 'lightgoldenrodyellow', label = 'Training data')
        plt.scatter(y_test_pred, y_test_pred - y_test, c = 'tomato', label = 'Test data')
        plt.hlines(y = 0, xmin = -10, xmax = 250, lw = 2, color = 'snow')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.legend(loc = 'best')
        plt.show()

    def ransac_regression(feature, target):
        '''
            Function for fitting data to robust linear estimation using RANSAC algorithm
                Args:
                    feature variable, target variable
                _____________
                Returns:
                    ❅ figure
        '''
        ## Arranging data ##
        x = df[feature].values.reshape(-1, 1)
        y = df[target].values
        ## Instantiating the algorithm ##
        ransac = RANSACRegressor()
        ## Fitting the model ##
        ransac.fit(x, y)
        ## Adding inlier_mask ##
        inlier = ransac.inlier_mask_
        outlier = np.logical_not(inlier)
        ## Adjusting x-axis to plot lines ##
        line_x = np.arange(x.min(),x.max())[:,np.newaxis]
        line_y_ransac = ransac.predict(line_x.reshape(-1, 1))
        ## Plotting figure ##
        plt.figure(figsize = (10, 8))
        plt.scatter(x[inlier], y[inlier], c = 'darkorange', marker = 'o', label = 'Inliers')
        plt.scatter(x[outlier], y[outlier], c = 'darkblue', marker = "*", label = 'Outliers')
        plt.plot(line_x, line_y_ransac, color = 'navajowhite')
        plt.legend(loc = 'best')
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.show()

    def robust_regression(feature, target):
        '''
            Function to robustly fit a linear model to faulty data using the RANSAC algorithm.
                Args:
                    feature variable, target variable
                ____________
                Returns:
                    ❅ float: Estimated Coefficients
                    ❅ figure
        '''
        ## Arranging data ##
        x = df[feature].values.reshape(-1, 1)
        y = df[target].values
        ## Instantiating the model ##
        model = LinearRegression()
        model.fit(x, y)
        ## Instantiating the RANSAC algorithm ##
        ransac = RANSACRegressor()
        ransac.fit(x, y)
        ## Adding inlier_mask ##
        inlier = ransac.inlier_mask_
        outlier = np.logical_not(inlier)
        ## Adjusting x-axis to plot lines ##
        line_x = np.arange(x.min(),x.max())[:,np.newaxis]
        line_y = model.predict(line_x)
        line_y_ransac = ransac.predict(line_x.reshape(-1, 1))
        ## Printing the coefficients ##
        print('Estimated Coefficients(Linear Regression, RANSAC Regression):')
        print(model.coef_, ransac.estimator_.coef_)
        ## Plotting the figure ##
        lw = 2
        plt.figure(figsize = (10, 8))
        plt.scatter(x[inlier], y[inlier], c = 'darkorange', marker = 'o', label = 'Inliers')
        plt.scatter(x[outlier], y[outlier], c = 'darkblue', marker = "*", label = 'Outliers')
        plt.plot(line_x, line_y, color='red', linewidth=lw, label='Linear regressor')
        plt.plot(line_x, line_y_ransac, color = 'navajowhite', linewidth = lw, label ='RANSAC Regressor')
        plt.legend(loc = 'best')
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.show()
        
    def multiple_regression(features, target):
        '''
            Function for fitting model with Multiple Linear Regression using statsmodels.api module
                Args:
                    pd.Series or list
                _______________
                Returns:
                    pd.DataFrame: Ordinary Least Squares Regression Results
        '''
        ## Arranging data ##
        x = df[features]
        y = df[target]
        ## Adding the constant ##
        x_constant = sm.add_constant(x)
        pd.DataFrame(x_constant)
        ## Fit and summary ##
        model = sm.OLS(y, x_constant)
        lr = model.fit()
        lr.summary()
        ## Predicted Values ##
        print('Predicted Values:', lr.predict())
        return lr.summary()

    def polynomial_transformation(feature, target, degree):
        '''
            Function to generate polynomial and interaction features.
                Args:
                    feature variable, target variable, degree of transformation
                _______________
                Returns:
                    ❅ float: Coefficient of Determination
                    ❅ figure
        '''
        ## Arranging data ##
        x = df[feature].values
        y = df[target].values
        ## Instantiating the algorithms ##
        poly = PolynomialFeatures(degree=degree)
        lr = LinearRegression()
        ## Fitting and transforming the data ##
        poly_x1 = poly.fit_transform(x.reshape(-1, 1))
        lr.fit(poly_x1, y.reshape(-1, 1))
        ## Making predictions ##
        y_pred1 = lr.predict(poly_x1)
        ## Plotting the figure ##
        plt.figure(figsize = (15, 8))
        plt.scatter(x, y)
        plt.plot(x, y_pred1, color = 'gold')
        plt.title('Polynomial Transformation with Linear Regression')
        print('R² Score:{:.4f}'.format(r2_score(y, y_pred1)))
        plt.show()

    def decision_tree(feature, target, depth):
        '''
            Function for fitting data to a decision tree regressor
                Args: 
                    feature variable, target variable, maximum depth of the tree
                ______________
                Returns:
                    ❅ figure: Decision Tree Regressor
        '''
        ## Instantiating the algorithm ##
        tree = DecisionTreeRegressor(max_depth = depth)
        ## Arranging the data ##
        x = df[[feature]].values
        y = df[[target]].values
        ## Fitting the data ##
        tree.fit(x, y)
        ## Sorting the data ##
        sort_idx = x.flatten().argsort()
        ## Plotting the data ##
        plt.figure(figsize = (10, 8))
        plt.scatter(x[sort_idx], y[sort_idx])
        plt.plot(x[sort_idx], tree.predict(x[sort_idx]), color = 'gold')
        plt.xlabel(feature)
        plt.ylabel(target)
        return tree.predict(x[sort_idx])

    def train_test(features, labels, target, size, model = str, ratio = None, depth = None):
        """
            Function for preprocessing the data, splitting data into training and test data and calculate the accuracy of the model

            Parameters:
                features: features for predicting target variable
                labels:   Categorical values that require preprocessing
                target:   target variables
                size:     Test size after splitting data
                model:    Machine learning algorithm
                ratio:    valid only if model is ElasticNet 
                          The ElasticNet mixing parameter
                depth:    depth of the tree. Valid only if model = tree and ada
            ______________________

            Returns:
                Mean Squared Error of the training and the test data
                Coefficent of Determination R² of the training and the test data
                Residual Analysis figure
        """
        ## Data Preprocessing ##
        column_trans = make_column_transformer(
        (OneHotEncoder(), labels),remainder = 'passthrough')
        X = df[features]
        y = df[target]
        x = column_trans.fit_transform(X)
        ## Splitting data for training and testing
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = size, random_state = 0)
        ## Instantiating the model ##
        if model == 'linear':
            model = LinearRegression()
        elif model == 'ransac':
            model = RANSACRegressor()
        elif model == 'tree':
            model = DecisionTreeRegressor(max_depth=depth)
        elif model == 'elastic':
            model = ElasticNet(l1_ratio=ratio, alpha = 1, max_iter=100)
        elif model == 'ridge':
            model = Ridge()
        elif model == 'ada':
            model = AdaBoostRegressor(DecisionTreeRegressor(max_depth = depth),
                           n_estimators = 500, random_state = 42)
        elif model == 'forest':
            model = RandomForestRegressor(n_estimators=500, criterion='mse',
                                   random_state=42, n_jobs=1)
        ## fitting the model ##
        model.fit(x_train, y_train)
        ## prediction results ##
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        ## Printing the MSE and the R² score ##
        print('Mean Squared Error: Training Data %.2f' %(mean_squared_error(y_train, y_train_pred)))
        print('Mean Squared Error: Test Data %.2f' %(mean_squared_error(y_test, y_test_pred)))
        print('Coefficient of Determination R²: Training Data %.2f' %(r2_score(y_train, y_train_pred)))
        print('Coefficient of Determination R²: Test Data %.2f' %(r2_score(y_test, y_test_pred)))
        ## Residual Analysis ##
        plt.figure(figsize = (10, 8))
        plt.scatter(y_train_pred, y_train_pred - y_train, c = 'lightgoldenrodyellow', label = 'Training data')
        plt.scatter(y_test_pred, y_test_pred - y_test, c = 'tomato', label = 'Test data')
        plt.hlines(y = 0, xmin = -10, xmax = 250, lw = 2, color = 'snow')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.legend(loc = 'best')
        plt.show()

    def learning_curve(features, labels, target, estimator, title, ylim = None, cv = None, n_jobs = 1, train_sizes = np.linspace(.1, 1.0, 5)):
        '''
            Function to determine cross-validated training and test scores for different training set sizes
                Args:
                    features: list or columns
                    labels : list; columns with categorical values
                    target: pd.Series or str; target variable
                    estimator: model algorithm
                    cv: Cross validation set
            ______________________
            Returns:
                ❅ Train and Test mean scores
                ❅ Learning Curves Figure
        '''
        column_trans = make_column_transformer(
        (OneHotEncoder(), labels),remainder = 'passthrough')
        X = df[features]
        y = df[target]
        x = column_trans.fit_transform(X)

        plt.figure(figsize = (10, 8))
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, x, y, cv = cv, n_jobs = n_jobs, train_sizes =train_sizes)
        train_scores_mean = np.mean(train_scores, axis = 1)
        train_scores_std = np.std(train_scores, axis = 1)
        test_scores_mean = np.mean(test_scores, axis = 1)
        test_scores_std = np.std(test_scores, axis = 1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha = 0.1, color = 'r')
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha = 0.1, color = 'g')
        plt.plot(train_sizes, train_scores_mean, 'o-', color = 'r', label = 'Training score')
        plt.plot(train_sizes, test_scores_mean, 'o-', color = 'g', label = 'Cross Validation Score')
        plt.legend(loc = 'best')
        plt.show()

        vtrain = print('Training %.4f'.format(train_scores_mean))
        vtest = print('Test %.4f'.format(test_scores_mean))

        return vtrain, vtest