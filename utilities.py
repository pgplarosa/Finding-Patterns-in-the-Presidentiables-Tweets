# python standard libraries
import re
import sys
from datetime import datetime
import time
from collections import Counter
import itertools
from IPython.core.display import HTML
from IPython.display import display

# data processing
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from tqdm import tqdm
tqdm.pandas()

# plotting
from pandas.plotting import scatter_matrix
from matplotlib import cm
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
import pylab as plot
import matplotlib.pyplot as plt

# preprocessing
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, auc, f1_score
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import average_precision_score


# normalize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import ConfusionMatrixDisplay

# ml models
# error based models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

# information based models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# similarity based
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

# scoring metrics
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# dimentionality reduction 
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

# naive bayes
from sklearn.naive_bayes import GaussianNB

# resampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as imbpipeline

# model interpretability
import shap
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
import eli5
from eli5.sklearn import PermutationImportance
import lime
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_text import LimeTextExplainer

# time series
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# text data
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# Download nltk requirements
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
import advertools as adv

from wordcloud import WordCloud
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer

# ignore warnings
import warnings
from sklearn.exceptions import ConvergenceWarning


candidates = ['lenirobredo', 'bongbongmarcos', 'IskoMoreno', 
              'SAPBongGo', 'iampinglacson']
models = [LinearSVC(C=0.1), LogisticRegression(C=0.1, dual=False),  
          LogisticRegression(C=1, dual=False), 
          LinearSVC(C=0.1), LogisticRegression(penalty='l1', C=1,
                                              solver='liblinear')]
cols_exclude = [['lang', 'yung', 'medical', 'medical center'],
                ['upang', 'nating', 'natin', 'nasa', 'maraming', 
                 'nang', 'ilang'],
                ['natin', 'january'],
                [],
                ['still' , 'like']]


def preprocess_text(df):
    # remove urls
    url = df.text.str.replace(r'https?://\S+', '')
    
    # tokenize
    tokenize = url.apply(nltk.word_tokenize)

    # casefold
    lower_case = tokenize.apply(lambda x: 
                               list(map(lambda y: y.casefold(), x)))
    
    
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatize = lower_case.apply(lambda x: list(map(lemmatizer.lemmatize, 
                                                             x)))

    # remove stopwords
    stop_words = (list(set(stopwords.words('english'))) 
                  + sorted(adv.stopwords['tagalog']))
    filtered_stopwords = lemmatize.apply(lambda x: 
                                         list(filter(lambda y:y not 
                                                     in stop_words, 
                                                     x)))

    # filter words with less than 3 character length
    filtered_words = filtered_stopwords.apply(lambda x: 
                                             list(filter(lambda y:len(y) > 3,
                                                         x)))

    return filtered_words

# extract data
def get_feat_targ(df, candidate_name, qcut=3, ngram_start=1):
    df_cand = df.loc[(df.username == candidate_name)].copy()
    df_cand['total_engagements'] = df_cand.iloc[:, 4:-1].sum(axis=1)
    df_cand = pd.concat([df_cand.iloc[:, -2:], df_cand.iloc[:, 3]], axis=1) 
    
    filtered_words = preprocess_text(df_cand)
    df_cand['clean_tokenize'] = filtered_words
    df_cand['clean'] = df_cand.clean_tokenize.apply(' '.join)
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'\b[a-z]+\b', 
                                       ngram_range=(ngram_start, 2),
                                       max_df=0.8,
                                       min_df=0.01)

    bow = tfidf_vectorizer.fit_transform(df_cand['clean'])
    df_cand_bow = pd.DataFrame.sparse.from_spmatrix(
                        bow, columns=tfidf_vectorizer.get_feature_names_out())

    df_cand['binned'] = pd.Series(pd.qcut(df_cand.total_engagements, 2))
    df_cand['binned'] = [1 if i == df_cand.binned.max() 
                         else 0 for i in df_cand['binned']]
    
    return df_cand_bow, df_cand.binned, tfidf_vectorizer

def train(X, y):
    C = [ 1e-3, .01, 0.1, 1, 10, 100, 1000]
    n_neighbors = list(range(1, 51))
    max_depth = list(range(1, 10))
    resample = RandomOverSampler(random_state=143)

    automl_tree = AutoML(['dtc', 'rfc', 'gbc'])
    splits = automl_tree.split_data(X, y, shuffle=True, num_trials=10, 
                                    test_size=0.25)
    automl_tree.train_model(X, y, param_grid={'max_depth':max_depth},
                            plot_train_val=False, 
                            plot_feat_imp=False)
    tree_summary = automl_tree.generate_summary()

    automl_linear_l1 = AutoML(['log', 'svl'])
    splits = automl_linear_l1.split_data(X, y, shuffle=True, num_trials=10, 
                                         test_size=0.25)
    automl_linear_l1.train_model(X, y, param_grid= {'C' : C}, 
                                fixed_params={'penalty' : 'l1'},
                                plot_train_val=False, 
                                plot_feat_imp=False)
    linearl1_summary = automl_linear_l1.generate_summary()

    automl_linear_l2 = AutoML(['log', 'svl'])
    splits = automl_linear_l2.split_data(X, y, shuffle=True, 
                                         num_trials=10, 
                                         test_size=0.25)
    automl_linear_l2.train_model(X, y, param_grid= {'C' : C}, 
                                fixed_params={'penalty' : 'l2'},
                                plot_train_val=False, 
                                plot_feat_imp=False)
    linearl2_summary = automl_linear_l2.generate_summary()


    df = pd.concat([tree_summary, 
                    linearl1_summary, 
                    linearl2_summary]).reset_index(drop=True)
    
    return df


class AutoML(object):
    def __init__(self, model_code):
        self.splits = []
        models = {
            # regressor
            # error based
            'lir' : [LinearRegression, 'Linear Regression'],
            'rdg' : [Ridge, 'Ridge Regression'],
            'lso' : [Lasso, 'Lasso Regression'],
            # information based
            'dtr' : [DecisionTreeRegressor, 'Decision Tree'], 
            'rfr' : [RandomForestRegressor, 'Random Forest'], 
            'gbr' : [GradientBoostingRegressor, 'GradientBoost'],
            
            'knr' : [KNeighborsRegressor, 'knn Regression'],

            # classifier
            # error based
            'log' : [LogisticRegression, 'Logistic Regression'],
            'svl' : [LinearSVC, 'Linear SVM'],
            'svm' : [SVC, 'SVM'],
            
            # information based
            'dtc' : [DecisionTreeClassifier, 'Decision Tree'],
            'rfc' : [RandomForestClassifier, 'Random Forest'],
            'gbc' : [GradientBoostingClassifier, 'Gradient Boost'], 
            # similarity based
            'knc' : [KNeighborsClassifier, 'knn Classification'],    
        }
        # multiple train
        self.summs = None
        self.test_model = None
        if isinstance(model_code, list):
            if all(map(lambda x:x in models, model_code)):
                self.model_code = model_code
                self.model = [models[i][0] for i in model_code]
                self.model_name = [models[i][1] for i in model_code]
            else:
                raise ValueError('invalid model_code!')
        else:
            if model_code not in models.keys():
                raise ValueError('invalid model_code!')
            self.model_code = model_code
            self.model = models[model_code][0]
            self.model_name = models[model_code][1]
        # used only for appending for model_name
        self.append = True
        
    def split_data(self, X, y, random_state=1337, shuffle=True, 
                   method='train_test', num_trials=1, stratify=False, 
                   **kwargs):
        stratify_dict = {}
        if self.splits:
            return self.splits
        
        if method == 'train_test':
            if stratify:
                stratify_dict['stratify'] = y
                
            for i in range(num_trials):
                test_size = (0.2 if 'test_size' not in kwargs.keys() 
                             else kwargs['test_size'])

                (X_train, X_test, 
                 y_train, y_test) = train_test_split(X, y,
                                                    test_size=test_size,
                                                    random_state=i, 
                                                    shuffle=shuffle,
                                                    **stratify_dict)
                
                self.splits.append((X_train, X_test, y_train, y_test))
        elif method == 'kfold':
            n_splits = (8 if 'n_splits' not in kwargs.keys() 
                        else kwargs['n_splits'])
            if stratify:
                kf = StratifiedKFold(n_splits=n_splits, 
                                     shuffle=shuffle, 
                                     random_state=random_state)
                self.splits = [(X.iloc[i], X.iloc[j], y.iloc[i], y.iloc[j]) 
                               for (i, j) in kf.split(X, y)]
            else:
                kf = KFold(n_splits=n_splits, shuffle=shuffle, 
                           random_state=random_state)
                self.splits = [(X.iloc[i], X.iloc[j], y.iloc[i], y.iloc[j]) 
                               for (i, j) in kf.split(X)]

        return self.splits
    
    def generate_summary(self,):
        if self.summs is not None:
            return self.summs
        return [self.model_name, self.test_score, self.best_param, 
                self.top_predictor, self.run_time]
    
    def plot_feature_importance(self, absol=True, sort=True):
        fig, ax = plt.subplots(figsize=(6,8))
        mean_coefs = self.mean_coefs.copy()
        feature_names = self.feature_names.copy()

        if absol:
            mean_coefs = np.abs(mean_coefs)
        
        if sort:
            feature_names = feature_names[np.argsort(np.abs(mean_coefs))]
            mean_coefs = sorted(np.abs(mean_coefs))
        ax.set_title(self.model_name)
        ax.barh(np.arange(self.coefs_count), mean_coefs)
        ax.set_yticks(np.arange(self.coefs_count))
        ax.set_yticklabels(feature_names)
    
    def plot_train_val(self, log_scale=False):
        # Initialize figure
        fig, ax = plt.subplots(figsize=(15, 6))
        plt.title(self.model_name, fontsize=16)
        values = [list(param.values())[0] for param in self.param_grid]
        
        if log_scale:
            plt.xscale('log')
            
        # Plot the spread of predictions
        ax.fill_between(
            values,
            np.mean(self.score_train, axis=0) + np.std(self.score_train, 
                                                       axis=0),
            np.mean(self.score_train, axis=0) - np.std(self.score_train, 
                                                       axis=0),
            color='tab:blue', alpha=0.10
        )
        ax.fill_between(
            values,
            np.mean(self.score_test, axis=0) + np.std(self.score_test, 
                                                      axis=0),
            np.mean(self.score_test, axis=0) - np.std(self.score_test, 
                                                      axis=0),
            color='tab:orange', alpha=0.10
        )
        # Plot mean of the predictions
        ax.plot(values, np.mean(self.score_train, axis=0), lw=3,
                label="training accuracy")
        ax.plot(values, np.mean(self.score_test, axis=0), lw=3,
                label="validation accuracy")
        
        plt.xlabel(list(self.param_grid[0].keys())[0])
        plt.legend()
        plt.ylabel('accuracy')
        
    def set_model_default_params(self, model):
        default_params = {
            'rfr' : {'n_jobs' : -1, 'n_estimators':50}, 
            'rfc' : {'n_jobs' : -1, 'n_estimators':50}, 
            'gbr' : {'n_estimators':50},
            'gbc' : {'n_estimators':50}
        }
        
        if self.model_code in default_params.keys():
            model.set_params(**default_params[self.model_code])
        
        elif self.model_code == 'log':
            if ('penalty' in self.fixed_params.keys() 
                and self.fixed_params['penalty'] == 'l1'):
                default_params = {'solver' : 'liblinear'}
            else:
                default_params = {'dual' : False}
        
            model.set_params(**default_params)
            if self.append:
                self.model_name = f'{self.model_name} ({self.fixed_params["penalty"]})'  
                self.append = False
        
        elif self.model_code == 'svl':
            if ('penalty' in self.fixed_params.keys() 
                and self.fixed_params['penalty'] == 'l1'):
                default_params = {'loss' : 'squared_hinge', 'dual' : False}
                model.set_params(**default_params)
            if self.append:
                self.model_name = f'{self.model_name} ({self.fixed_params["penalty"]})'  
                self.append = False
        
        return model
    
    def get_coefs(self, reg):
        if self.model_code in ['dtr', 'dtc', 
                               'rfr', 'rfc', 
                               'gbr', 'gbc']:
            coefs = reg.feature_importances_
        elif self.model_code in ['lir', 'rdg', 'lso', 'log', 'svl']:
            coefs = reg.coef_
        elif self.model_code in ['knr', 'knc']:
            coefs = None

        return coefs
    
    def train_model(self, X, y,
                    random_state=0, fixed_params={}, 
                    param_grid={'max_depth':range(2,11)}, 
                    normalize=False, plot_train_val=True, 
                    plot_feat_imp=True, resample=None):
        if isinstance(self.model_code, list):
            self.train_models_multiple(X, y,
                    random_state, fixed_params, 
                    param_grid, 
                    normalize, plot_train_val, 
                    plot_feat_imp, resample)
            return 
        
        start_time = time.time()
        score_train = []
        score_test = []
        weighted_coefs=[]
        param_grid = list(ParameterGrid(param_grid))
        model = self.model
        self.fixed_params = fixed_params
        self.normalize = normalize
        
        with tqdm(total=len(self.splits), file=sys.stdout, 
                  position=0, leave=True) as pbar:
            for split in self.splits:
                pbar.set_description(self.model_name)
                
                training_accuracy = []  
                test_accuracy = []
                X_train, X_test, y_train, y_test = split

                if normalize: 
                    # normalize the features
                    Scaler = normalize().fit(X_train)
                    X_train = Scaler.transform(X_train)
                    X_test = Scaler.transform(X_test)
                
                if resample:
                    (X_train, 
                     y_train) = resample.fit_resample(X_train.to_numpy(), 
                                                      y_train)
                    
                if self.model_code in ['lir']:
                    mdl = model()
                    mdl.fit(X_train, y_train)
                    training_accuracy.append(mdl.score(X_train, y_train))
                    test_accuracy.append(mdl.score(X_test, y_test))
                    coefs = self.get_coefs(mdl)
                    weighted_coefs.append(coefs)
                else:
                    for param in param_grid:
                        try:
                            mdl = model(random_state=random_state, 
                                         **param)
                        except TypeError:
                            mdl = model(**param)

                        mdl = self.set_model_default_params(mdl)
                        try:
                            mdl.set_params(**fixed_params)
                        except KeyError:
                            pass

                        mdl.fit(X_train, y_train)

                        training_accuracy.append(mdl.score(X_train, y_train))
                        test_accuracy.append(mdl.score(X_test, y_test))

                        coefs = self.get_coefs(mdl)
                        weighted_coefs.append(coefs)
                pbar.update(1)

                score_train.append(training_accuracy)
                score_test.append(test_accuracy)

        # get the mean of the weighted coefficients over all the trials  
        mean_coefs = (None if self.model_code in ['knr', 'knc'] 
                      else np.mean(weighted_coefs, axis=0))
        mean_coefs = (None if self.model_code in ['knr', 'knc'] 
                      else (mean_coefs if mean_coefs.ndim == 1 
                            else mean_coefs.mean(axis=0)))
        score = np.mean(score_test, axis=0)
        top_predictor = ('NA' if self.model_code in ['knr', 'knc'] 
                        else X.columns[np.argmax(np.abs(mean_coefs))])
        
        # for plotting train and test accuracy
        self.score_train = score_train
        self.score_test = score_test
        self.param_grid = param_grid
        
        # for plotting feature importance
        self.mean_coefs = (None if self.model_code in ['knr', 'knc'] 
                           else mean_coefs)
        self.coefs_count = (None if self.model_code in ['knr', 'knc'] 
                            else len(self.mean_coefs) )
        self.feature_names = X.columns
        
        # for summary
        self.test_score = np.amax(score)
        best_param = [f'{key} =  {value}' 
                      for key, value 
                      in param_grid[np.argmax(score)].items()]
        self.best_param = (';'.join(best_param) if self.model_code != 'lir' 
                           else 'NA')
        self.top_predictor = top_predictor
        self.run_time = (time.time() - start_time)
        
        # for score
        self.test_param = param_grid[np.argmax(score)]
        
        if plot_train_val:
            if self.model_code == 'lir':
                pass
            elif self.model_code not in ['log', 'svl']:
                self.plot_train_val()
            else:
                self.plot_train_val(log_scale=True)
        
        if plot_feat_imp:
            if self.model_code not in ['knr', 'knc']:
                self.plot_feature_importance()
    
    def train_models_multiple(self, X, y,
                              random_state=0, fixed_params={}, 
                              param_grid={'max_depth':range(2,11)}, 
                              normalize=False, plot_train_val=True, 
                              plot_feat_imp=True, resample=None):
        """should have the same params"""
        model_codes = self.model_code
        models = self.model
        model_names = self.model_name
        summs = []
        for model_code, model, model_name in zip(model_codes, models, 
                                                 model_names):
            self.append = True
            self.model_code = model_code
            self.model = model
            self.model_name = model_name
            self.train_model(X, y, random_state, fixed_params, 
                            param_grid, 
                            normalize, plot_train_val, 
                            plot_feat_imp, resample=resample)
            summs.append(self.generate_summary())
            
        
        summs = pd.DataFrame(summs)
        summs.columns = ['Machine Learning Method', 'Test Accuracy', 
                         'Best Parameter', 'Top Predictor Variable', 
                         'Run Time']
        
        self.summs = summs
    
    def score(self, X, y):
        """ Works for single model only"""
        if self.test_model is None:
            X_train, X_test, y_train, y_test = self.splits[0]
            
            if self.normalize: 
                # normalize the features
                Scaler = self.normalize().fit(X_train)
                X_train = Scaler.transform(X_train)
                X_test = Scaler.transform(X_test)
                self.test_scaler = Scaler
            
            test_model = self.model()
            test_model = self.set_model_default_params(test_model)
            try:
                test_model.set_params(**self.fixed_params)
            except KeyError:
                pass

            test_model.set_params(**self.test_param)
            test_model.fit(X_train, y_train)
            self.test_model = test_model
        
        if self.normalize:
            X = self.test_scaler.transform(X)
        
        return self.test_model.score(X, y)
    

def word_cloud_candidate(df):
    df_cand = df.groupby('username')['text'].apply(lambda x: ' '.join(x)).copy()

    display(HTML(f'<h3><center>Word Cloud per Candidate</center></h3>'))
    fig, ax = plt.subplots(3, 2, figsize=(15, 15))
    ax = ax.flatten()
    for i, cand in enumerate(df_cand.index):
        text = df_cand.iloc[i]
        generated_wc = (WordCloud(background_color='black', 
                                  width=800, height=400,
                                  )
                        .generate(text))
    
        ax[i].set_title(cand)
        ax[i].imshow(generated_wc,
                     interpolation="bilinear"
                    )
    
    
    ax[5].set_visible(False)
    

def shap_wordcloud(model, candidate, df, cols_exc, low, high):
    username_map = {
        'lenirobredo' : 'Leni Robredo',
        'bongbongmarcos' : 'Bongbong Marcos',
        'IskoMoreno' : 'Isko Moreno',
        'SAPBongGo' : 'Bong Go', 
        'iampinglacson': 'Ping Lacson'
    }
    X, y, tfidf_vectorizer = get_feat_targ(df, candidate, 
                                           qcut=2, ngram_start=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=143,
                                                        shuffle=True,
                                                        stratify=y)
    cols_exclude = [X_test.columns.get_loc(c) for c in cols_exc 
                    if c in X_test]
    cols_index = list(set(range(len(X_test.columns))) - set(cols_exclude))

    model = model
    model.fit(X_train, y_train)
    display(HTML(f'<h2><center>{username_map[candidate]}</center></h2>'))
    acc = round(model.score(X_test, y_test) * 100, 2) 
    display(HTML(f'<h3><center>Accuracy: {acc}%</center><h3>'))
    
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)

    plt.xticks(rotation=90)
    plt.show();
    y_test_pred = model.predict(X_test)
    
    
    explainer = shap.LinearExplainer(model, X_test)
    # generate shap values
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values[:, cols_index], X_test.iloc[:, cols_index],
                      show=False, plot_type='violin', plot_size=(8,8) 
                      )

    shap_values = shap_values[:, cols_index]

    plt.title('High Engagement')
    plt.show()
    
    
    
    cols = X_test.iloc[:, cols_index].columns

    shap_posi = np.where((np.abs(np.min(shap_values, axis=0)) 
                          < np.abs(np.max(shap_values, axis=0))),
                         np.abs(
                             np.sum(np.where(shap_values > 0, shap_values, 0), 
                                    axis=0)),
                         0)
    shap_nega = np.where((np.abs(np.min(shap_values, axis=0)) 
                          > np.abs(np.max(shap_values, axis=0))),
                         np.abs(
                             np.sum(np.where(shap_values < 0, shap_values, 0), 
                                    axis=0)),
                         0)

    mult_posi = np.round((shap_posi)*1000).astype(int)
    mult_nega = np.round((shap_nega)*1000).astype(int)

    fig, ax = plt.subplots(1, 2, figsize=(15, 10))

    weights_high = {i: j for i, j in zip(cols, mult_posi)}
    wc = WordCloud(background_color='white', colormap='Greens')
    wc.generate_from_frequencies(weights_high)
    ax[0].imshow(wc, interpolation='bilinear')
    ax[0].axis("off")
    ax[0].set_title('High engagement', fontdict={'fontsize': 20, 
                                                'fontweight': 'bold'})
    
    weights_low = {i: j for i, j in zip(cols, mult_nega)}
    wc = WordCloud(background_color='white', colormap='Reds')
    wc.generate_from_frequencies(weights_low)
    ax[1].imshow(wc, interpolation='bilinear')
    ax[1].axis("off")
    ax[1].set_title('Low engagement', fontdict={'fontsize': 20, 
                                                'fontweight': 'bold'})
    fig.tight_layout()
    plt.show()
    
    display(HTML(f'<h3><center>Sample Tweet: Low Engagement</center></h3>'))
    low = df[(df.username==candidate) & 
             (df.text.str.contains(low, re.IGNORECASE))].text.iloc[0]
    model = LogisticRegression(C=0.1, dual=False)
    model.fit(X.to_numpy(), y)
    c = make_pipeline(tfidf_vectorizer, model)

    explainer = LimeTextExplainer(class_names=['Low Engagement',
                                               'High Engagement'])
    exp = explainer.explain_instance(low, 
                                     c.predict_proba, top_labels=0,
                                     num_features =5)
    exp.show_in_notebook(text=low)
    
    display(HTML(f'<h3><center>Sample Tweet: High Engagement</center></h3>'))
    high = df[(df.username==candidate) & 
              (df.text.str.contains(high, re.IGNORECASE))].text.iloc[0]
    model = LogisticRegression(C=0.1, dual=False)
    model.fit(X.to_numpy(), y)
    c = make_pipeline(tfidf_vectorizer, model)

    explainer = LimeTextExplainer(class_names=['Low Engagement',
                                               'High Engagement'])
    exp = explainer.explain_instance(high, 
                                     c.predict_proba, top_labels=1,
                                     num_features =5)
    exp.show_in_notebook(text=high)
    

def multi_class_preprocessing(df, title=False):
    """Returns the X and y features that will be used for the multi-class model"""
    df_all = df.copy()
    filtered_words = preprocess_text(df_all)
    
    df_all['clean_tokenize'] = filtered_words
    df_all['clean'] = df_all.clean_tokenize.apply(' '.join)

    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'\b[a-z]+\b', 
                                   ngram_range=(1, 2),
                                   max_df=0.8,
                                   min_df=0.01)

    bow = tfidf_vectorizer.fit_transform(df_all['clean'])
    df_all_bow = pd.DataFrame.sparse.from_spmatrix(
                    bow, columns=tfidf_vectorizer.get_feature_names_out())
    
    X = df_all_bow
    le = LabelEncoder()
    y = le.fit_transform(df_all.username)
    name = le.classes_
    if title:
        return name
    else: 
        return X, y
    

def multi_class_model(X, y):
    """Returns a dataframe on the results of the multi-class model"""
    C = [ 1e-3, .01, 0.1, 1, 10, 100, 1000]
    n_neighbors = list(range(1, 51))
    max_depth = list(range(1, 10))
    rs = None

    # automl_tree = AutoML(['dtc', 'rfc', 'gbc']) # long running yung gbc. un comment and edit
    automl_tree = AutoML(['dtc', 'rfc'])
    splits = automl_tree.split_data(X, y, shuffle=True, num_trials=10, test_size=0.25)
    automl_tree.train_model(X, y, param_grid={'max_depth':max_depth}, resample = rs)
    tree_summary = automl_tree.generate_summary()

    automl_linear_l1 = AutoML(['log', 'svl'])
    splits = automl_linear_l1.split_data(X, y, shuffle=True, num_trials=10, test_size=0.25)
    automl_linear_l1.train_model(X, y, param_grid= {'C' : C}, 
                        fixed_params={'penalty' : 'l1'}, resample = rs)
    linearl1_summary = automl_linear_l1.generate_summary()

    automl_linear_l2 = AutoML(['log', 'svl'])
    splits = automl_linear_l2.split_data(X, y, shuffle=True, num_trials=10, test_size=0.25)
    automl_linear_l2.train_model(X, y, param_grid= {'C' : C}, 
                        fixed_params={'penalty' : 'l2'}, resample = rs)
    linearl2_summary = automl_linear_l2.generate_summary()

    model_df = pd.concat([tree_summary, 
                linearl1_summary, 
                linearl2_summary]).reset_index(drop=True)
    return model_df


def multi_class_viz(X, y, df, i, conf_matrix=False):
    """Returns a shap or confusion matrix visualization for model interpretability"""
    name = multi_class_preprocessing(df, title=True)
    
    # retrain using the selected model and using the whole train set
    model = LogisticRegression(C = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=143, 
                                                        shuffle=True,
                                                        stratify=y)
    model.fit(X_train.to_numpy(), y_train)
    
    
    if conf_matrix:
        ConfusionMatrixDisplay.from_estimator(model, X_test.to_numpy(), y_test,
                                            display_labels=name)
        plt.xticks(rotation=90)
        plt.show()

        plt.show();
        y_test_pred = model.predict(X_test.to_numpy())
        cm = confusion_matrix(y_test, y_test_pred)

        accurate=0
        for j in range(len(name)):
            recall_i = recall_score(y_test, y_test_pred, average=None)[j]*100
            print("Test Recall for candidate {0:s}: {1:.2f}%".format(name[j],
                                            recall_i))
        print('\nOverall Test Accuracy:'
              ' {0:.4f}%'.format(accuracy_score(y_test,
                                            y_test_pred,
                                            normalize=True)*100))
    
    else:
        explainer = shap.LinearExplainer(model, X_test)

        # generate shap values
        shap_values = explainer.shap_values(X_test)
    
        shap.summary_plot(shap_values[i], X_test, show=False)
        plt.title(name[i])
        plt.show()
        

def multi_class_engagement_model():
    df = pd.read_csv('candidate_tweetsv2.csv')
    df_all = df.copy()
    df_all = df_all[df_all.username.isin(['lenirobredo', 'bongbongmarcos', 'IskoMoreno', 'SAPBongGo', 'iampinglacson'])]
    df_all = df_all.reset_index(drop=True)

    filtered_words = preprocess_text(df_all)
    df_all['clean_tokenize'] = filtered_words
    df_all['clean'] = df_all.clean_tokenize.apply(' '.join)

    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'\b[a-z]+\b', 
                                       ngram_range=(1, 2),
                                       max_df=0.8,
                                       min_df=0.01)

    bow = tfidf_vectorizer.fit_transform(df_all['clean'])
    df_all_bow = pd.DataFrame.sparse.from_spmatrix(
                        bow, columns=tfidf_vectorizer.get_feature_names_out())


    X = pd.concat([df_all_bow, df_all.iloc[:,4:8]], axis=1)
    le = LabelEncoder()
    y = le.fit_transform(df_all.username)
    
    C = [ 1e-3, .01, 0.1, 1, 10, 100, 1000]
    n_neighbors = list(range(1, 51))
    max_depth = list(range(1, 10))
    rs = RandomOverSampler(random_state=143)

    automl_tree = AutoML(['dtc', 'rfc', 'gbc'])
    splits = automl_tree.split_data(X, y, shuffle=True, num_trials=10, test_size=0.25)
    automl_tree.train_model(X, y, param_grid={'max_depth':max_depth}, resample = rs)
    tree_summary = automl_tree.generate_summary()

    automl_linear_l1 = AutoML(['log', 'svl'])
    splits = automl_linear_l1.split_data(X, y, shuffle=True, num_trials=10, test_size=0.25)
    automl_linear_l1.train_model(X, y, param_grid= {'C' : C}, 
                            fixed_params={'penalty' : 'l1'}, resample = rs)
    linearl1_summary = automl_linear_l1.generate_summary()

    automl_linear_l2 = AutoML(['log', 'svl'])
    splits = automl_linear_l2.split_data(X, y, shuffle=True, num_trials=10, test_size=0.25)
    automl_linear_l2.train_model(X, y, param_grid= {'C' : C}, 
                            fixed_params={'penalty' : 'l2'}, resample = rs)
    linearl2_summary = automl_linear_l2.generate_summary()
    
    df = pd.concat([tree_summary, 
                    linearl1_summary, 
                    linearl2_summary]).reset_index(drop=True)
    return df


def tweet_count(df):
    """Returns a plot count of tweets per candidate"""
    df.groupby(['username'])['id'].count().plot(kind='bar')
    plt.xlabel('Candidate')
    plt.show()
    

def tweet_distribution(df, cand_name):
    display(HTML(f'<b>{cand_name}</b>'))
    plt.subplots(figsize=(10, 2))
    df_candidate = df.loc[df.username==cand_name].copy()
    df_candidate['created_at'] = pd.to_datetime(df_candidate.created_at)
    df_candidate['total_engagement'] = df_candidate.iloc[:, 4:8].sum(axis=1)
    print('Count of Tweets:', df_candidate['created_at'].count())
    print('Earliest Tweet:',
          df_candidate['created_at'].min().strftime('%B %d, %Y'))
    print('Latest Tweet:',
          df_candidate['created_at'].max().strftime('%B %d, %Y'))
    display(df_candidate['total_engagement'].describe().apply("{0:,.0f}".format))
    plt.title('Tweets Over Time')
    plt.bar(df_candidate.created_at, df_candidate.total_engagement)
    plt.tight_layout()
    plt.show()