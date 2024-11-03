import nltk
from nltk import edit_distance
from nltk.translate.bleu_score import SmoothingFunction
# nltk.download('wordnet')

import re
from collections import deque
from typing import List, Optional, Tuple

from apted import APTED, Config
from apted.helpers import Tree
from Levenshtein import distance, ratio
from lxml import etree, html
from lxml.html import HtmlElement
import numpy as np

from scipy import stats

# KTD calculation
def normalised_kendall_tau_distance(values1, values2):
    """Compute the Kendall tau distance."""
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(values1)
    b = np.argsort(values2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered / (n * (n - 1))

# calculate order metrics at word level
def compute_word_order(pred, gt, minlen=1):
    metrics = dict()

    gt = gt.split()
    pred = pred.split()
    if len(gt) <= minlen: 
        return metrics
    
    def get_common_unique_elements(list1, list2):
        # co-occuring elements
        common_elements = set(list1) & set(list2)
        
        # construct two sub-list
        unique_list1 = []
        unique_list2 = []
        
        for word in list1:
            if word in common_elements and word not in unique_list1:
                unique_list1.append(word)
        
        for word in list2:
            if word in common_elements and word not in unique_list2:
                unique_list2.append(word)
        
        word_2_id = dict()
        for idx, word in enumerate(unique_list1):
            word_2_id[word] = idx

        unique_list1 = [word_2_id[word] for word in unique_list1]
        unique_list2 = [word_2_id[word] for word in unique_list2]

        return unique_list1, unique_list2
    
    unique_gt, unique_pred = get_common_unique_elements(gt, pred)
    if len(unique_gt) > 0.1 * min(len(gt), len(pred)) and len(unique_gt) >= 2:
        metrics['kendall_tau'] = 1 - normalised_kendall_tau_distance(unique_gt, unique_pred) 
        metrics['spearmanr'] = stats.spearmanr(unique_gt, unique_pred)[0]
    else:
        metrics['kendall_tau'] = 0
        metrics['spearmanr'] = 0

    return metrics

# calculate order metrics at segment level
def compute_segment_order(pred_segments, gt_segments, minlen=0):
    metrics = dict()

    if len(gt_segments) <= minlen: 
        return metrics
    
    def get_common_unique_elements(list1, list2):
        # construct two new sub-list
        idx_list1 = []
        idx_list2 = []
        for segment_2 in list2: # gold segment
            min_score = 1
            min_idx = -1
            for idx, segment_1 in enumerate(list1):
                if idx in idx_list1:
                    continue
                edit_dist = distance(segment_2, segment_1) / max(len(segment_2), len(segment_1))
                if edit_dist <= 0.5 and edit_dist < min_score:
                    min_idx = idx
                    min_score = edit_dist
            if min_idx != -1:
                idx_list1.append(min_idx)
                idx_list2.append(len(idx_list2))
        idx_list1 = list(np.argsort(idx_list1))

        return idx_list1, idx_list2
    
    unique_pred, unique_gt = get_common_unique_elements(pred_segments, gt_segments)
    if len(unique_gt) > 0.1 * min(len(gt_segments), len(pred_segments)) and len(unique_gt) >= 2:
        metrics['kendall_tau'] = 1 - normalised_kendall_tau_distance(unique_gt, unique_pred)
        metrics['spearmanr'] = stats.spearmanr(unique_gt, unique_pred)[0]
    else:
        metrics['kendall_tau'] = 0
        metrics['spearmanr'] = 0

    return metrics

# compute concat metrics, e.g. concat EDS for text, tables ...
def compute_concated_metrics(pred, gt, minlen=0):
    metrics = dict()

    if len(gt) <= minlen:  # not evaluation if len(gt) less than minlen
        return metrics
    
    metrics["edit_dist"] = distance(pred, gt) / max(len(pred), len(gt))
    metrics['edit_dist_sim'] = 1 - metrics["edit_dist"]
    
    reference = gt.split()
    hypothesis = pred.split()
    metrics["bleu"] = nltk.translate.bleu([reference], hypothesis, smoothing_function=SmoothingFunction().method1)  # 增加Smooth方法
    metrics["meteor"] = nltk.translate.meteor([reference], hypothesis)

    reference = set(reference)
    hypothesis = set(hypothesis)
    metrics["precision"] = nltk.scores.precision(reference, hypothesis)
    if metrics['precision'] is None:
        metrics['precision'] = 0
    metrics["recall"] = nltk.scores.recall(reference, hypothesis)
    if metrics['recall'] is None:
        metrics['recall'] = 0
    metrics["f_measure"] = nltk.scores.f_measure(reference, hypothesis)
    if metrics['f_measure'] is None:
        metrics['f_measure'] = 0

    return metrics


# calculate table teds
class TableTEDS:
    '''Tree Edit Distance basead Similarity'''

    def __init__(
        self, structure_only: bool = False, ignore_nodes: Optional[str] = None
    ):
        self.structure_only = structure_only
        self.ignore_nodes = ignore_nodes
        self.__tokens__: List[str] = []

    def __call__(self, pred: str, gt: str) -> float:
        """Computes TEDS score between the prediction and the ground truth of a
        given sample
        Args:
            pred (str): The predict html string of the table image.
            gt (str): The ground truth html string of the table image.
        Returns:
            float: TEDS score
        """
        if (not pred) or (not gt):
            return 0.0

        parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
        pred_element: HtmlElement = html.fromstring(pred, parser=parser)
        gt_element: HtmlElement = html.fromstring(gt, parser=parser)

        xpath_ele = 'body/table'
        if pred_element.xpath(xpath_ele) and gt_element.xpath(xpath_ele):
            pred_element = pred_element.xpath(xpath_ele)[0]
            gt_element = gt_element.xpath(xpath_ele)[0]

            if self.ignore_nodes:
                etree.strip_tags(pred_element, *self.ignore_nodes)
                etree.strip_tags(gt_element, *self.ignore_nodes)

            tree_pred = self.load_html_tree(pred_element)
            tree_true = self.load_html_tree(gt_element)
            distance = APTED(
                tree_pred, tree_true, TedsTableConfig()
            ).compute_edit_distance()

            n_nodes_pred = len(pred_element.xpath(".//*"))
            n_nodes_true = len(gt_element.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            # n_nodes = n_nodes_pred + n_nodes_true
            return 1.0 - (float(distance) / n_nodes)
        return 0.0

    def tokenize(self, node: HtmlElement) -> None:
        '''Tokenizes table cells'''
        self.__tokens__.append(f'<{node.tag}>')

        if node.text is not None:
            self.__tokens__ += list(node.text)

        for n in node.getchildren():
            self.tokenize(n)

        if node.tag != 'unk':
            self.__tokens__.append(f'</{node.tag}>')

        if node.tag != 'td' and node.tag != 'th' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(
        self, node: HtmlElement, parent: Optional[HtmlElement] = None
    ) -> Optional[HtmlElement]:
        '''Converts HTML tree to the format required by apted'''
        global __tokens__
        if node.tag == 'td' or node.tag == 'th':
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()

            new_node = TedsTableTree(
                node.tag,
                int(node.attrib.get('colspan', '1')),
                int(node.attrib.get('rowspan', '1')),
                cell,
                *deque(),
            )
        else:
            new_node = TedsTableTree(node.tag, None, None, None, *deque())

        if parent is not None:
            parent.children.append(new_node)

        if node.tag != 'td' and node.tag != 'th':
            for n in node.getchildren():
                self.load_html_tree(n, new_node)

        if parent is None:
            return new_node
        return None


# construct table as a tree structure
class TedsTableTree(Tree):
    def __init__(
        self,
        tag: str,
        colspan: Optional[int] = None,
        rowspan: Optional[int] = None,
        content: Optional[List[str]] = None,
        *children: Tuple[Tree],
    ) -> None:
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)


class TedsTableConfig(Config):
    def rename(self, node1: TedsTableTree, node2: TedsTableTree) -> float:
        """Compares attributes of trees"""
        if (
            (node1.tag != node2.tag)
            or (node1.colspan != node2.colspan)
            or (node1.rowspan != node2.rowspan)
        ):
            return 1.0

        if node1.tag == 'td' or node1.tag == 'th':
            if node1.content or node2.content:
                max_len = max(len(node1.content), len(node2.content))
                return distance(node1.content, node2.content) / max_len
        return 0.0


class TedsHeadNode:
    def __init__(self, content, depth, parent=None, children=None):
        self.content = content
        self.depth = depth
        self.parent = parent
        self.children = children if children else []


class TedsHeadConfig(Config):

    def rename(self, node1, node2):
        """node1:pred, node2:label"""
        if node1.content or node2.content:
            max_len = max(len(node1.content), len(node2.content))
            return distance(node1.content, node2.content) / max_len
            '''
            if distance(node1.content, node2.content) / max_len < 0.15:
                return 0.0
            return 1.0
            '''
        return 0.0


# calculate heading teds
def calc_head_teds(node_list_pred, node_list_gold):

    def build_head_tree(node_list):

        node_stack = [TedsHeadNode('Root', 0)]
        
        for node in node_list:
            node = node.strip()
            if re.match(r'^#+\s', node):
                label, title = node.split(' ', 1)
                level = len(label)
                title = title.strip()
            else:
                level = node.count('#')
                title = node.replace('#', '').strip()

            cur_node = TedsHeadNode(title, level)
            while level <= node_stack[-1].depth:
                node_stack.pop()
            cur_node.parent = node_stack[-1]
            node_stack[-1].children.append(cur_node)
            node_stack.append(cur_node)
            
        return node_stack[0]
    
    pred_tree = build_head_tree(node_list_pred)
    gold_tree = build_head_tree(node_list_gold)

    distance = APTED(pred_tree, gold_tree, TedsHeadConfig()).compute_edit_distance()
    teds = 1.0 - (float(distance) / max(len(node_list_gold), len(node_list_pred)))
    # teds = 1.0 - (float(distance) / (len(node_list_gold) + len(node_list_pred)))
    # teds = 1.0 - (float(distance) / max([len(node_list_gold) + 1, len(node_list_pred) + 1]))

    return teds