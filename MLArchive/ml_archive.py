import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

import os

class MLArchive:
    """MLArchive class to hold the training models history.

    This class should be used to record our models and 
    results from the iterative training process.
    
    """

    SCHEMA = [
        'id', 'technique', 'model', 'metric', 'date', 'train_res',
        'devel_res', 'test_res', 'params', 'train_samples',
        'test_samples', 'train_hist', 'columns', 'packages'
    ]

    def __init__(self, filename: str = None, models_path: str = 'archive/'):
        if filename == None:
            self.__model_path = models_path
            self.__ranked_models = pd.DataFrame(columns=self.SCHEMA)
        else:
            self.load_archive(filename)

    def save_model(self, mod: object, metric: str, train_res: float,
            test_res: float, devel_res: float = None,
            train_samples: int = None, test_samples: int = None,
            train_hist: dict = None, columns: list = None,
            packages: str = None) -> None:

        """
        This method saves a new entry for a new model and rank it 
        based on the test result metric.
        
        Parameters
        ----------
        mod : object
            The trained model.
        metric : str
            The metric which was used.
        train_res: float,
            Result of the chosen metric in the train set.
        test_res: float
            Result of the chosen metric in the test set.
        devel_res: float, default = None,
            Numer of train samples.
        train_samples: int, default = None
            Number of train samples.
        test_samples: int, default = None
            Number of test samples.
        train_hist: dict, default = None
            The training history per training step.
        columns: list, default = None
            The model input features.
        packages: str, default = None
            To preserve reciprocity could contain the model requirements.
        """

        model = {}
        model['model'] = mod
        model['metric'] = metric
        model['train_res'] = train_res
        model['devel_res'] = devel_res
        model['test_res'] = test_res
        model['train_samples'] = train_samples
        model['test_samples'] = test_samples
        model['train_hist'] = train_hist
        model['columns'] = columns
        model['packages'] = packages
        tech = ''
        if isinstance(mod, BaseEstimator):
            params = mod.get_params()
            name = type(mod).__name__
            if name == 'Pipeline':
                name = type(mod[-1]).__name__
            tech = name

        # TODO: Keras NN, XGBoost, LightGBM, ...
        elif isinstance(mod, None):
            tech = 'TODO' # TODO
        elif isinstance(mod, None):
            tech = 'TODO' # TODO
        elif isinstance(mod, None):
            tech = 'TODO' # TODO
        else:
            print("Sorry, we haven't implemented save for "\
                  "this kind of model. Please implement it on "\
                  "save_model and submit a pull request. Thanks!")
        
        now = datetime.now()
        model['params'] = params    #TODO convert this to json
        model['technique'] = tech
        model['date'] = now.strftime("%d/%m/%Y %H:%M:%S")
        model['id'] = now.strftime("%d%m%Y%H%M%S%f")[:-3]

        self.__update_rank(model)

    def __update_rank(self, model: dict) -> None:
        """ 
        This is a method to internally update our model archive and ranks.

        Parameters
        ----------
        model: dict
            model data to be saved.
        """
        df = self.__ranked_models
        df = df.append(model, ignore_index=True)

        df = df.sort_values(by='test_res', ascending=False)
        df = df.reset_index(drop=True)
        pos = str(df.loc[df['id']==model['id']].index[0])
        self.__ranked_models = df
        print('Model '+model['id']+' added in position: '+pos)

    def load_model(self, id: str) -> object:
        """ 
        This method load a previously trained model from the archive using 
        it's ID.

        Parameters
        ----------
        id: str
            ID of the model to be loaded.
        """
        # TODO

    def load_best_model(self) -> object:
        """
        Load a previously trained model with the top result in the test set.
        """
        return self.load_model(self.__ranked_models.iloc[0, 'id'])

    def get_ranked_models(self, lim: int = None, cols: list = range(8)
                          ) -> pd.DataFrame:
        """
        Retrieve the registry of trained models.
        
        Parameters
        ----------
        lim: int
            Number of registries to load.
        cols: Index or array-like
            Column labels to use for resulting frame.
        """
        return self.__ranked_models.iloc[:lim, cols]

    def get_path(self) -> str:
        """
        Get the path where we write the models.
        """
        return self.__model_path

    def save_archive(self, filename: str) -> None:
        """
        Write the archive to file.
        
        Parameters
        ----------
        filename: str
            File to write on.
        """
        folder = self.__model_path
        if not os.path.exists(folder):
            os.makedirs(folder)
        pickle.dump(self, open(folder+filename, "wb"))

    def load_archive(self, filename: str) -> None:
        """
        Load the archive from file.

        Parameters
        ----------
        filename: str
            File to load from.
        """
        data = pickle.load(open(filename, "rb"))
        self.__model_path = os.path.dirname(os.path.abspath(filename))
        self.__ranked_models = \
            data.get_ranked_models(cols = range(len(self.SCHEMA)))

    def plot_history(self, params: dict = None) -> None:
        """
        Plot the evolution of the project over time.

        Parameters
        ----------
        params: dict
            Plot custom parameters.
        """

        if params:
            if params['dark']:
                sns.set(style="ticks", context="talk")
                plt.style.use("dark_background")
                custom_style = {'axes.labelcolor': 'white',
                                'xtick.color': 'white',
                                'ytick.color': 'white'}
                sns.set_style("darkgrid", rc=custom_style)
        
        ax = sns.lineplot(x='date', y='test_res', data=self.__ranked_models)
        metric = self.__ranked_models['metric'].unique()
        ax.set_title(label='Model evolution: ' + metric + ' over time')
        return ax

    def plot_model_learning_curve(self, id: str, params: dict = None) -> None:
        """
        Plot the learning curve of the selected model.

        Parameters
        ----------
        id: str
            ID of the model.
        params: dict
            Plot custom parameters.
        """
        # TODO
