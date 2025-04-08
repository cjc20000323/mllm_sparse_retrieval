import torch.distributed as dist
import torch
from tqdm import tqdm


class RecallMetrics:

    def __init__(self, dataset, dense_run, sparse_run, fusion_run, look_up, lookup_indices, search_args):
        self.recall_k_setting_list = [1, 5, 10, 100, 200]
        self.dense_counts = {k: 0 for k in self.recall_k_setting_list}
        self.sparse_counts = {k: 0 for k in self.recall_k_setting_list}
        self.fusion_counts = {k: 0 for k in self.recall_k_setting_list}
        self.dense_recall_lists = {k: [[None] for _ in range(dist.get_world_size())] for k in
                                   self.recall_k_setting_list}
        self.sparse_recall_lists = {k: [[None] for _ in range(dist.get_world_size())] for k in
                                    self.recall_k_setting_list}
        self.fusion_recall_lists = {k: [[None] for _ in range(dist.get_world_size())] for k in
                                    self.recall_k_setting_list}

        self.dataset = dataset
        self.dense_run = dense_run
        self.sparse_run = sparse_run
        self.fusion_run = fusion_run

        self.look_up = look_up
        self.lookup_indices = lookup_indices
        self.search_args = search_args

    def _sort(self, dictionary):
        sorted_by_value = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
        sorted_by_value_dicts = {k: dict(sorted_by_value[:k]) for k in self.recall_k_setting_list}
        search_results = {k: list(sorted_by_value_dicts[k]) for k in self.recall_k_setting_list}
        search_results = {k: torch.tensor([int(i) for i in search_results[k]]).cuda() for k in
                          self.recall_k_setting_list}
        return search_results

    def sort_and_count(self):
        if len(self.dense_run) > 0:
            for k, v in tqdm(self.dense_run.items()):
                target = self.dataset.get_target(k, self.search_args.query_type)
                if isinstance(target, list):
                    target = torch.tensor([int(i) for i in target]).cuda()
                else:
                    target = int(target)
                if len(v['docs']) == 0:
                    continue

                search_results = self._sort(v['docs'])
                self._count('dense', search_results, target)
        if len(self.sparse_run) > 0:
            for k, v in tqdm(self.sparse_run.items()):
                target = self.dataset.get_target(k, self.search_args.query_type)
                if isinstance(target, list):
                    target = torch.tensor([int(i) for i in target]).cuda()
                else:
                    target = int(target)
                if len(v['docs']) == 0:
                    continue

                search_results = self._sort(v['docs'])
                self._count('sparse', search_results, target)

        if len(self.fusion_run) > 0:
            for k, v in tqdm(self.fusion_run.items()):
                target = self.dataset.get_target(k, self.search_args.query_type)
                if isinstance(target, list):
                    target = torch.tensor([int(i) for i in target]).cuda()
                else:
                    target = int(target)
                if len(v) == 0:
                    continue

                search_results = self._sort(v)
                self._count('fusion', search_results, target)

    def _count(self, result_type, search_results, target):
        for k in search_results:
            if True in torch.isin(search_results[k], target):
                if result_type == 'dense':
                    self.dense_counts[k] += 1
                elif result_type == 'sparse':
                    self.sparse_counts[k] += 1
                else:
                    self.fusion_counts[k] += 1

    def all_gather_object(self):
        self.dense_counts = {k: self.dense_counts[k] / (len(self.lookup_indices) * dist.get_world_size()) for k in
                             self.recall_k_setting_list}
        for k in self.recall_k_setting_list:
            dist.all_gather_object(object_list=self.dense_recall_lists[k], obj=self.dense_counts[k])

        self.sparse_counts = {k: self.sparse_counts[k] / (len(self.lookup_indices) * dist.get_world_size()) for k in
                              self.recall_k_setting_list}
        for k in self.recall_k_setting_list:
            dist.all_gather_object(object_list=self.sparse_recall_lists[k], obj=self.sparse_counts[k])

        self.fusion_counts = {k: self.fusion_counts[k] / (len(self.lookup_indices) * dist.get_world_size()) for k in
                              self.recall_k_setting_list}
        for k in self.recall_k_setting_list:
            dist.all_gather_object(object_list=self.fusion_recall_lists[k], obj=self.fusion_counts[k])

    def print_recall(self):
        if dist.get_rank() == 0:
            print(len(self.lookup_indices) * dist.get_world_size())
            if len(self.dense_run) > 0:
                print(len(self.look_up))
                dense_recalls = {k: sum(self.dense_recall_lists[k]) for k in self.recall_k_setting_list}
                for k in self.recall_k_setting_list:
                    print('Dense recall @ {}: {}'.format(k, self.dense_recall_lists[k]))
                print(
                    'Dense reps recall: r@1 {}, r@5 {}, r@10 {}, r@100 {}, r@200 {}'.format(dense_recalls[1],
                                                                                            dense_recalls[5],
                                                                                            dense_recalls[10],
                                                                                            dense_recalls[100],
                                                                                            dense_recalls[200]))

            if len(self.sparse_run) > 0:
                sparse_recalls = {k: sum(self.sparse_recall_lists[k]) for k in self.recall_k_setting_list}
                for k in self.recall_k_setting_list:
                    print('Sparse recall @ {}: {}'.format(k, self.sparse_recall_lists[k]))
                print(
                    'Sparse reps recall: r@1 {}, r@5 {}, r@10 {}, r@100 {}, r@200 {}'.format(sparse_recalls[1],
                                                                                             sparse_recalls[5],
                                                                                             sparse_recalls[10],
                                                                                             sparse_recalls[100],
                                                                                             sparse_recalls[200]))
            if len(self.fusion_run) > 0:
                fusion_recalls = {k: sum(self.fusion_recall_lists[k]) for k in self.recall_k_setting_list}
                for k in self.recall_k_setting_list:
                    print('Fusion/Hybrid recall @ {}: {}'.format(k, self.fusion_recall_lists[k]))
                print(
                    'Fusion/Hybrid reps recall: r@1 {}, r@5 {}, r@10 {}, r@100 {}, r@200 {}'.format(fusion_recalls[1],
                                                                                                    fusion_recalls[5],
                                                                                                    fusion_recalls[10],
                                                                                                    fusion_recalls[100],
                                                                                                    fusion_recalls[
                                                                                                        200]))
