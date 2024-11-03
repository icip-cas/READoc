from typing import List, Dict

import numpy as np
import pandoc
from scipy.optimize import linear_sum_assignment

from utils import compute_concated_metrics, TableTEDS, calc_head_teds, compute_segment_order, compute_word_order


# table recognition
class TableEvaluate():

    def __init__(self, table_list_gold : List, table_list_pred : List):
        
        self.gold_tables = table_list_gold
        self.pred_tables = table_list_pred

        # default using latex tables

        gold_table_concated = "\n".join(self.gold_tables).strip()
        pred_table_concated = "\n".join(self.pred_tables).strip()
        self.concated_metrics = compute_concated_metrics(pred_table_concated, gold_table_concated)

        # compute TEDS metrics
        self.html_gold_tables = []
        self.html_pred_tables = []

        try:
            for table in self.gold_tables:
                html_table = pandoc.write(pandoc.read(table, format='latex'), format="html")
                html_table = '<html><body>' + html_table + '</body></html>'
                self.html_gold_tables.append(html_table)
            if len(self.html_gold_tables) == 0:
                self.mapped_metrics = dict()
            else:
                self.mapped_metrics = {"teds_p" : None, "teds_r" : None, "teds_f1" : None}
        except:
            self.mapped_metrics = dict()
        
        if len(self.mapped_metrics.keys()) > 0:
            for table in self.pred_tables:
                try:
                    html_table = pandoc.write(pandoc.read(table, format='latex'), format="html")
                    html_table = '<html><body>' + html_table + '</body></html>'
                except:
                    html_table = ""   # empty -> 0
                
                self.html_pred_tables.append(html_table)
            
            teds_module = TableTEDS()
            distance_matrix = np.zeros((len(self.gold_tables), len(self.pred_tables)))
            for i in range(len(self.gold_tables)):
                for j in range(len(self.pred_tables)):
                    distance_matrix[i][j] = teds_module(self.html_gold_tables[i], self.html_pred_tables[j])
            
            row_ind, col_ind = linear_sum_assignment(distance_matrix, maximize = True)
            teds_sum = distance_matrix[row_ind, col_ind].sum()
            
            if len(self.html_pred_tables) > 0:
                self.mapped_metrics['teds_p'] = teds_sum / len(self.html_pred_tables)
            else:
                self.mapped_metrics['teds_p'] = 0
            self.mapped_metrics['teds_r'] = teds_sum / len(self.html_gold_tables)
            self.mapped_metrics['teds_min'] = min(self.mapped_metrics['teds_p'], self.mapped_metrics['teds_r'])
            if (self.mapped_metrics['teds_p'] + self.mapped_metrics['teds_r']) == 0:
                self.mapped_metrics['teds_f1'] = 0
            else:
                self.mapped_metrics['teds_f1'] = 2 * self.mapped_metrics['teds_p'] * self.mapped_metrics['teds_r'] / (self.mapped_metrics['teds_p'] + self.mapped_metrics['teds_r'])

        self.metric_dict = {'concated': self.concated_metrics, 'mapped': self.mapped_metrics}

# formula conversion                
class MathEvaluate():
    def __init__(self, math_gold : Dict, math_pred : Dict):

        self.gold_inline_maths = math_gold['inline']
        self.gold_outline_maths = math_gold['outline']
        self.pred_inline_maths = math_pred['inline']
        self.pred_outline_maths = math_pred['outline']

        self.gold_inline_concated = "\n".join(self.gold_inline_maths).strip()
        self.gold_outline_concated = "\n".join(self.gold_outline_maths).strip()

        self.pred_inline_concated = "\n".join(self.pred_inline_maths).strip()
        self.pred_outline_concated = "\n".join(self.pred_outline_maths).strip()

        self.inline_concated_metrics = compute_concated_metrics(self.pred_inline_concated, self.gold_inline_concated)
        self.outline_concated_metrics = compute_concated_metrics(self.pred_outline_concated, self.gold_outline_concated)

        self.metric_dict = {'inline_concated' : self.inline_concated_metrics, 'outline_concated': self.outline_concated_metrics}

# text extraction
class TextEvaluate():
    def __init__(self, text_gold : Dict, text_pred : Dict):

        self.text_gold = text_gold
        self.text_pred = text_pred

        self.metrics = compute_concated_metrics(self.text_pred.strip(), self.text_gold.strip())

        self.metric_dict = {'metrics' : self.metrics}

# heading detection
class HeadEvaluate():
    def __init__(self, head_gold : List, head_pred : List):
        self.gold_heads = head_gold
        self.pred_heads = head_pred

        if len(self.gold_heads) == 0:
            self.concated_metrics = dict()
            self.logical_metrics = dict()
            
        else:
        
            gold_head_concated = "\n".join(self.gold_heads).strip()
            pred_head_concated = "\n".join(self.pred_heads).strip()
            self.concated_metrics = compute_concated_metrics(pred_head_concated, gold_head_concated)

            self.logical_metrics = {'teds': calc_head_teds(head_pred, head_gold)}

        self.metric_dict = {'concated': self.concated_metrics, 'logical': self.logical_metrics}

# reading order detection
class OrderEvaluate():
    def __init__(self, md_gold, md_pred, segments_gold, segments_pred):
        self.segment_order = compute_segment_order(segments_pred, segments_gold)
        self.word_order = compute_word_order(md_pred, md_gold)
        self.metric_dict = {'segment': self.segment_order, 'word': self.word_order}

