import collections
import pandas as pd
from helpers import get_performance, get_score, subtotal, get_recall, get_auc
import copy
import math

PRE, REC, SPEC, FPR, NPV, ACC, F1 = 7, 6, 5, 4, 3, 2, 1
MATRIX = "\t".join(["\tTP", "FP", "TN", "FN"])
PERFORMANCE = " \t".join(["\tCLF", "PRE ", "REC", "SPE", "FPR", "NPV", "ACC", "F_1"])


class FFT(object):
    def __init__(self,criteria ,target,max_level=1):
        self.max_depth = max_level - 1
        cnt = 2 ** self.max_depth
        self.tree_cnt = cnt
        self.tree_depths = [0] * cnt
        self.best = -1

        self.target = target
        self.criteria = criteria

        self.train = None
        self.test = None

        self.structures = None
        self.computed_cache = {}

        self.selected = [{} for _ in range(cnt)]
        self.tree_scores = [None] * cnt
        self.node_descriptions = [None] * cnt
        self.performance_on_train = [collections.defaultdict(dict) for _ in range(cnt)]
        self.performance_on_test = [None] * cnt
        self.predictions = [None] * cnt
        self.loc_aucs = [None] * cnt

    "Build all possible tress."

    def build_trees(self):
        self.structures = self.get_all_structure()
        data = copy.deepcopy(self.train)
        for i in range(self.tree_cnt):
            self.grow(data, i, 0, [0, 0, 0, 0])

    "Evaluate all tress built on TEST data."

    def eval_trees(self):
        for i in range(self.tree_cnt):
            # Get performance on TEST data.
            self.eval_tree(i)

    "Find the best tree based on the score."

    def find_best_tree(self):
        goals = ['recall','precision','pf']
        if self.tree_scores and self.tree_scores[0]:
            return
        if not self.performance_on_test or not self.performance_on_test[0]:
            self.eval_trees()
        print("\t----- PERFORMANCES FOR ALL FFTs on Training Data -----")
        print(PERFORMANCE + " \t" + self.criteria)
        best = [-1, float('inf')]
        all_score = {}
        _id = 0
        for i in range(self.tree_cnt):
            # all_metrics = self.performance_on_test[i]
            all_metrics = self.performance_on_train[i][self.tree_depths[i]]
            if self.criteria == "LOC_AUC":
                score = self.loc_aucs[i]
            else:
                score = get_score(self.criteria, all_metrics[:4])
            self.tree_scores[i] = score
            print("\t" + "\t".join(
                ["FFT(" + str(i) + ")"] + \
                [str(x).ljust(5, "0") for x in all_metrics[4:] + [score]]))
            all_score[_id] = score
            _id += 1
        _temp = []
        for key in all_score.keys():
            _temp.append(all_score[key])
        _temp_df = pd.DataFrame(_temp, columns = ['precision','recall','pf'])
        dom_score = []
        for row_id in range(_temp_df.shape[0]):
            project_name = _temp_df.iloc[row_id].name
            row = _temp_df.iloc[row_id].tolist()
            wins = self.dominate(_temp_df,row,project_name,goals)
            dom_score.append(wins)
        _temp_df['wins'] = dom_score
        best = [_temp_df.wins.idxmax(),all_score[_temp_df.wins.idxmax()]]
        print("\tThe best tree found on training data is: FFT(" + str(best[0]) + ")")
        self.best = best[0]
        print(self.print_tree(best[0]))
        print(self.selected)
        return self.performance_on_test[best[0]][4:]

    "Given how the decision is made, get the description for the node."

    def describe_decision(self, t_id, level, metrics, reversed=False,_from="None"):
        cue, direction, threshold, decision = self.selected[t_id][level]
        #print('inside describe_decision', _from)
        tp, fp, tn, fn = metrics
        results = ["\'Good\'", "\'Bug!\'"]
        description = ("\t| " * (level + 1) + \
                       " ".join([cue, direction, str(threshold)]) + \
                       "\t--> " + results[1 - decision if reversed else decision]).ljust(30, " ")
        pos = "\tFalse Alarm: " + str(fp) + ", Hit: " + str(tp)
        neg = "\tCorrect Rej: " + str(tn) + ", Miss: " + str(fn)
        if not reversed:
            description += pos if decision == 1 else neg
        else:
            description += neg if decision == 1 else pos
        return description

    "Given how the decision is made, get the performance for this decision."

    def eval_decision(self, data, cue, direction, threshold, decision):
        try:
            if direction == ">":
                pos, neg = data.loc[data[cue] > threshold], data.loc[data[cue] <= threshold]
            else:
                pos, neg = data.loc[data[cue] < threshold], data.loc[data[cue] >= threshold]
        except:
            return 1, 2, 3
        if decision == 1:
            undecided = neg
        else:
            pos, neg = neg, pos
            undecided = pos
        # get auc for loc.
        sorted_data = pd.concat([df.sort_values(by=["LOC"], ascending=[1]) for df in [pos, neg]])
        #loc_auc = get_auc(sorted_data)
        tp = pos.loc[pos[self.target] == 1]
        fp = pos.loc[pos[self.target] == 0]
        tn = neg.loc[neg[self.target] == 0]
        fn = neg.loc[neg[self.target] == 1]
        # pre, rec, spec, fpr, npv, acc, f1 = get_performance([tp, fp, tn, fn])
        # return undecided, [tp, fp, tn, fn, pre, rec, spec, fpr, npv, acc, f1]
        return undecided, map(len, [tp, fp, tn, fn])#, loc_auc

    "Evaluate the performance of the given tree on the TEST data."

    def eval_tree(self, t_id):
        if self.performance_on_test[t_id]:
            return
        depth = self.tree_depths[t_id]
        self.node_descriptions[t_id] = [[] for _ in range(depth + 1)]
        TP, FP, TN, FN = 0, 0, 0, 0
        data = self.test
        for level in range(depth + 1):
            print(level)
            cue, direction, threshold, decision = self.selected[t_id][level]
            undecided, metrics = self.eval_decision(data, cue, direction, threshold, decision)
            _metrics = copy.deepcopy(metrics)
            _metrics2 = copy.deepcopy(metrics)
            tp, fp, tn, fn = self.update_metrics(level, depth, decision, metrics,'eval_tree')
            description = self.describe_decision(t_id, level, _metrics,False,"Eval_tree")
            #description = 'For Now'
            self.node_descriptions[t_id][level] += [description]
            TP, FP, TN, FN = TP + tp, FP + fp, TN + tn, FN + fn
            if len(undecided) == 0:
                break
            data = undecided
        description = self.describe_decision(t_id, level, _metrics2, reversed=True,_from = "Eval_tree")
        #description = 'For Now'
        self.node_descriptions[t_id][level] += [description]

        pre, rec, spec, fpr, npv, acc, f1 = get_performance([TP, FP, TN, FN])
        self.performance_on_test[t_id] = [TP, FP, TN, FN, pre, rec, spec, fpr, npv, acc, f1]
        return self.performance_on_test[t_id]

    def eval_other_project(self, t_id):
        depth = self.tree_depths[t_id]
        self.node_descriptions[t_id] = [[] for _ in range(depth + 1)]
        TP, FP, TN, FN = 0, 0, 0, 0
        data = self.test
        for level in range(depth + 1):
            print(level)
            cue, direction, threshold, decision = self.selected[t_id][level]
            undecided, metrics = self.eval_decision(data, cue, direction, threshold, decision)
            _metrics = copy.deepcopy(metrics)
            _metrics2 = copy.deepcopy(metrics)
            tp, fp, tn, fn = self.update_metrics(level, depth, decision, metrics,'eval_tree')
            description = self.describe_decision(t_id, level, _metrics,False,"Eval_tree")
            #description = 'For Now'
            self.node_descriptions[t_id][level] += [description]
            TP, FP, TN, FN = TP + tp, FP + fp, TN + tn, FN + fn
            if len(undecided) == 0:
                break
            data = undecided
        description = self.describe_decision(t_id, level, _metrics2, reversed=True,_from = "Eval_tree")
        #description = 'For Now'
        self.node_descriptions[t_id][level] += [description]

        pre, rec, spec, fpr, npv, acc, f1 = get_performance([TP, FP, TN, FN])
        self.performance_on_test[t_id] = [TP, FP, TN, FN, pre, rec, spec, fpr, npv, acc, f1]
        return self.performance_on_test[t_id]


    def predict(self, data, t_id=-1):
        # predictions = pd.Series([None] * len(data))
        if t_id == -1:
            t_id = self.best
        original = data
        original['prediction'] = pd.Series([None] * len(data))
        depth = self.tree_depths[t_id]
        for level in range(depth + 1):
            cue, direction, threshold, decision = self.selected[t_id][level]
            undecided, metrics = self.eval_decision(data, cue, direction, threshold, decision)
            decided_idx = [i for i in data.index if i not in undecided.index]
            original['prediction'][decided_idx] = decision
            # original.iloc[data.index, 'prediction'] = decision
            data = undecided
        # original.iloc[data.index, 'prediction'] = 1 if decision == 0 else 0
        original['prediction'][undecided.index] = 1 if decision == 0 else 0
        if None in original['prediction']:
            print("ERROR!")
        self.predictions[t_id] = original['prediction'].values
        return self.predictions[t_id]


    "Grow the t_id_th tree for the level with the given data"

    def grow(self, data, t_id, level, cur_performance):
        """
        :param data: current data for future tree growth
        :param t_id: tree id
        :param level: level id
        :return: None
        """
        goals = ['recall','precision','pf']
        if level >= self.max_depth:
            return
        if len(data) == 0:
            print("?????????????????????? Early Ends ???????????????????????")
            return
        self.tree_depths[t_id] = level
        decision = self.structures[t_id][level]
        structure = tuple(self.structures[t_id][:level + 1])
        cur_selected = self.computed_cache.get(structure, None)
        TP, FP, TN, FN = cur_performance
        all_score = {}
        _id = 0
        if not cur_selected:
            for cue in list(data):
                if cue == self.target:
                    continue
                threshold = data[cue].median()
                for direction in "><":
                    undecided, metrics = self.eval_decision(data, cue, direction, threshold, decision)
                    tp, fp, tn, fn = self.update_metrics(level, self.max_depth, decision, metrics)
                    if self.criteria == "cdom":
                        score = get_score(self.criteria, [TP + tp, FP + fp, TN + tn, FN + fn])
                    else:
                        score = get_score(self.criteria, [TP + tp, FP + fp, TN + tn, FN + fn])
                    
                    all_score[_id] = {'rule': (cue, direction, threshold, decision), \
                                        'undecided': undecided, \
                                        'metrics': [TP + tp, FP + fp, TN + tn, FN + fn], \
                                        'score': score}
                    _id += 1
            _temp = []
            for key in all_score.keys():
                _temp.append(all_score[key]['score'])
            _temp_df = pd.DataFrame(_temp, columns = ['precision','recall','pf'])
            dom_score = []
            for row_id in range(_temp_df.shape[0]):
                project_name = _temp_df.iloc[row_id].name
                row = _temp_df.iloc[row_id].tolist()
                wins = self.dominate(_temp_df,row,project_name,goals)
                dom_score.append(wins)
            _temp_df['wins'] = dom_score
            cur_selected = all_score[_temp_df.wins.idxmax()]
            self.computed_cache[structure] = cur_selected
        self.selected[t_id][level] = cur_selected['rule']
        self.performance_on_train[t_id][level] = cur_selected['metrics'] + get_performance(cur_selected['metrics'])
        self.grow(cur_selected['undecided'], t_id, level + 1, cur_selected['metrics'])

    "Given tree id, print the specific tree and its performances."

    def print_tree1(self, t_id):
        depth = self.tree_depths[t_id]
        for i in range(depth + 1):
            print(self.node_descriptions[t_id][i][0])
        print(self.node_descriptions[t_id][i][1])

        print("\t----- CONFUSION MATRIX -----")
        print(MATRIX)
        print("\t" + "\t".join(map(str, self.performance_on_test[t_id][:4])))

        print("\t----- PERFORMANCES ON TEST DATA -----")
        print(PERFORMANCE + " \t" + "Dist2Heaven")
        dist2heaven = get_score("Dist2Heaven", self.performance_on_test[t_id][:4])
        print("Performance:",self.performance_on_test[t_id])
        print("\t" + "\t".join(
            ["FFT(" + str(self.best) + ")"] + \
            [str(x).ljust(5, "0") for x in self.performance_on_test[t_id][4:] + [dist2heaven]]))
            # map(str, ["FFT(" + str(self.best) + ")"] + self.performance_on_test[t_id][4:] + [dist2heaven]))

    "Get all possible tree structure"

    def print_tree(self, t_id):
        data = self.test
        depth = self.tree_depths[t_id]
        string=''
        if not self.node_descriptions[t_id]:
            self.node_descriptions[t_id] = [[] for _ in range(depth + 1)]
        for i in range(depth + 1):
            if self.node_descriptions[t_id][i]:
                #print self.node_descriptions[t_id][i][0]
                string+=self.node_descriptions[t_id][i][0]+'\n'
            else:
                cue, direction, threshold, decision = self.selected[t_id][i]
                undecided, metrics = self.eval_decision(data, cue, direction, threshold, decision)
                description = self.describe_decision(t_id, i, metrics,False,"from print_tree")
                self.node_descriptions[t_id][i] += [description]
                #print description
                string+=description+'\n'
                if len(undecided) == 0:
                    break
                data = undecided
        return string

    def get_all_structure(self):
        def dfs(cur, n):
            if len(cur) == n:
                ans.append(cur)
                return
            dfs(cur + [1], n)
            dfs(cur + [0], n)

        if self.max_depth < 0:
            return []
        ans = []
        dfs([], self.max_depth)
        return ans

    "Update the metrics(TP, FP, TN, FN) based on the decision."

    def update_metrics(self, level, depth, decision, metrics,_from = 'simple'):
        tp, fp, tn, fn = metrics
        if level < depth:  # Except the last level, only part of the data(pos or neg) is decided.
            if decision == 1:
                tn, fn = 0, 0
            else:
                tp, fp = 0, 0
        return tp, fp, tn, fn


    def norm(self,x,df):
        lo = df.min()
        hi = df.max()
        return (x - lo) / (hi - lo +0.00000001)

    def dominate(self,_df,t,row_project_name,goals):
        wins = 0
        for i in range(_df.shape[0]):
            project_name = _df.iloc[i].name
            row = _df.iloc[i].tolist()
            if project_name != row_project_name:
                if self.dominationCompare(row, t,goals,_df):
                    wins += 1
        return wins

    def dominationCompare(self,other_row, t,goals,df):
        n = len(goals)
        weight = {'recall':1,'precision':1,'pf':-1}
        sum1, sum2 = 0,0
        for i in range(len(goals)):
            _df = df[goals[i]]
            w = weight[goals[i]]
            x = t[i]
            y = other_row[i]
            x = self.norm(x,_df)
            y = self.norm(y,_df)
            sum1 = sum1 - math.e**(w * (x-y)/n)
            sum2 = sum2 - math.e**(w * (y-x)/n)
        return sum1/n < sum2/n