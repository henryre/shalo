import numpy as np
import pandas as pd


class Hyperparameter(object):
    """Base class for a grid search parameter"""
    def __init__(self, name):
        self.name = name
    
    def get_all_values(self):
        raise NotImplementedError()
    
    def draw_values(self, n):
        # Multidim parameters can't use choice directly
        v = self.get_all_values()
        return [v[int(i)] for i in np.random.choice(len(v), n)]

    
class ListParameter(Hyperparameter):
    """List of parameter values for searching"""
    def __init__(self, name, parameter_list):
        self.parameter_list = np.array(parameter_list)
        super(ListParameter, self).__init__(name)
    
    def get_all_values(self):
        return self.parameter_list

    
class RangeParameter(Hyperparameter):
    """
    Range of parameter values for searching.
    min_value and max_value are the ends of the search range
    If log_base is specified, scale the search range in the log base
    step is range step size or exponent step size
    """
    def __init__(self, name, min_value, max_value, step=1, log_base=None):
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.log_base = log_base
        super(RangeParameter, self).__init__(name)
        
    def get_all_values(self):
        if self.log_base:
            min_exp = math.log(self.min_value, self.log_base)
            max_exp = math.log(self.max_value, self.log_base)
            exps = np.arange(min_exp, max_exp + self.step, step=self.step)
            return np.power(self.log_base, exps)
        return np.arange(
            self.min_value, self.max_value + self.step, step=self.step
        )
        

class GridSearch(object):
    """
    Runs hyperparameter grid search over a model object with train and score methods,
    training data (X), and training_marginals
    Selects based on maximizing F1 score on a supplied validation set
    Specify search space with Hyperparameter arguments
    """
    def __init__(self, model, train_data, train_labels, parameters):
        self.model              = model
        self.train_data         = train_data
        self.train_labels       = train_labels
        self.params             = parameters
        self.param_names        = [param.name for param in parameters]
        
    def search_space(self):
        return product(param.get_all_values() for param in self.params)

    def fit(self, dev_data, dev_labels, b=0.5, **model_hyperparams):
        """
        Basic method to start grid search, returns DataFrame table of results
          b specifies the positive class threshold for calculating accuracy
          Non-search parameters are set using model_hyperparamters
        """
        run_stats, score_opt, model_k = [], -1.0, 0
        base_model_name = self.model.name
        # Iterate over the param values
        for k, param_vals in enumerate(self.search_space()):
            model_name = '{0}_{1}'.format(base_model_name, model_k)
            model_k += 1
            # Set the new hyperparam configuration to test
            for pn, pv in zip(self.param_names, param_vals):
                model_hyperparams[pn] = pv
            print "=" * 80
            print "[%d] Testing %s" % (k+1, ', '.join([
                "%s = %0.2e" % (pn,pv)
                for pn,pv in zip(self.param_names, param_vals)
            ]))
            print "=" * 80
            # Train the model
            self.model.train(
                self.train_data, self.train_labels,
                dev_sentence_data=dev_data, dev_labels=dev_labels,
                **model_hyperparams
            )
            # Test the model
            score = self.model.score(dev_data, dev_labels, b=b, verbose=True)
            run_stats.append(list(param_vals) + [score])
            if score > score_opt:
                self.model.save(model_name)
                opt_model = model_name
                score_opt = score
        # Set optimal parameter in the learner model
        self.model.load(opt_model)
        # Return DataFrame of scores
        self.results = pd.DataFrame.from_records(
            run_stats, columns=self.param_names + ['Accuracy']
        ).sort_values(by='Accuracy', ascending=False)
        return self.results
    
    
class RandomSearch(GridSearch):
    def __init__(self, model, train_data, train_labels, parameters, n=10):
        """Search a random sample of size n from a parameter grid"""
        self.n = n
        super(RandomSearch, self).__init__(
            model, train_data, train_labels, parameters
        )
        print "Initialized RandomSearch search of size {0} / {1}".format(
            self.n, len(product(p.get_all_values() for p in self.params))
        )
        
    def search_space(self):
        return zip(*[param.draw_values(self.n) for param in self.params])
