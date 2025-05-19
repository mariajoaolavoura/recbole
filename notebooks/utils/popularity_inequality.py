import numpy as np
from recbole.evaluator.base_metric import AbstractMetric
from recbole.utils import EvaluatorType

class PopularityInequality(AbstractMetric):
    r'''PopularityInequality presents the diversity of the recommendation items.
    It is used to measure the inequality of a distribution.

    .. _GiniIndex: https://en.wikipedia.org/wiki/Gini_coefficient


    .. math::
        G = \frac{\sum_{i=1}^{n} \sum_{j=1}^{n} |x_i - x_j|}{2n \sum_{i=1}^{n} x_i}
        where x is the popularity of the item
        
    '''

    metric_type = EvaluatorType.RANKING
    smaller = True
    metric_need = ['rec.items', 'data.item_popularity']

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']

    def used_info(self, dataobject):
        '''Get the recommendation lists and the item popularity dictionary'''
        rec_lists = dataobject.get('rec.items')
        pop_dict = dataobject.get('data.item_popularity')
        return rec_lists, pop_dict


    def calculate_metric(self, dataobject):
        rec_lists, pop_dict = self.used_info(dataobject)
        metric_dict = {}
        for k in self.topk:
            key = '{}@{}'.format('popularityinequality', k)
            metric_dict[key] = np.round(
                self.get_popularity_inequality(rec_lists[:, :k], pop_dict), self.decimal_place
            )
        return metric_dict


    def calculate_metric_full_matrix(self, dataobject):
        rec_lists, pop_dict = self.used_info(dataobject)
        metric_dict = {}
        key = 'popularityinequality'
        metric_dict[key] = np.round(
            self.get_popularity_inequality(rec_lists, pop_dict), self.decimal_place
        )
        return metric_dict


    def get_popularity_inequality(self, rec_lists, pop_dict):
        '''Get popularity inequality index (Gini index) through the top-k recommendation list.

        Args:
            rec_lists(numpy.ndarray): 
            pop_dict(numpy.ndarray): 

        Returns:
            float: the popularity inequality index.
        '''

        # gini_index_list = []
        # for rec in rec_lists:

        #     n_rec_items = rec.size
        #     pop_rec_items = []
        #     pop_diff_list = []

        #     for i, item_i in enumerate(rec):
        #         pop_rec_items += [pop_dict[item_i]]

        #         for j, item_j in enumerate(rec):
        #             pop_diff_list += [abs(pop_dict[item_i] - pop_dict[item_j])]
                
            
        #     gini_index_list += [sum(pop_diff_list)/(2*n_rec_items*sum(pop_rec_items))]

        

        gini_index_list = []
        for rec in rec_lists:
            pop_rec_items = np.array([pop_dict[item] for item in rec])
            n = len(pop_rec_items)
            
            if n == 0 or pop_rec_items.sum() == 0:
                gini_index_list.append(0)
                continue

            # Compute pairwise absolute differences using broadcasting
            diff_matrix = np.abs(pop_rec_items[:, None] - pop_rec_items[None, :])
            # diff_matrix = pop_rec_items[:, None] - pop_rec_items[None, :]
            gini = diff_matrix.sum() / (2 * n * pop_rec_items.sum())

            gini_index_list.append(gini)

        
        return gini_index_list
