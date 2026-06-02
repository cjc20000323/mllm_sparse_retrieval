import torch.distributed as dist
import torch
from tqdm import tqdm
import pandas as pd


class RecallMetrics:

    def __init__(self, dataset, dense_run, sparse_run, fusion_run, look_up, lookup_indices, search_args):
        if dataset.data_name == 'fashion-iq':
            self.recall_k_setting_list = [1, 5, 10, 20, 30, 50, 100, 200, 300, 400]
        else:
            self.recall_k_setting_list = [1, 5, 10, 100, 200, 300, 400]

        self.fashion_iq_list = ['shirt', 'dress', 'toptee']
        self.fashion_iq_length = {'dress': 2017, 'shirt': 2038, 'toptee': 1961}

        if dataset.data_name == 'fashion-iq':
            self.dense_counts = {dress_type: {k: 0 for k in self.recall_k_setting_list} for dress_type in self.fashion_iq_list}
            self.sparse_counts = {dress_type: {k: 0 for k in self.recall_k_setting_list} for dress_type in self.fashion_iq_list}
            self.fusion_counts = {dress_type: {k: 0 for k in self.recall_k_setting_list} for dress_type in self.fashion_iq_list}
        else:
            self.dense_counts = {k: 0 for k in self.recall_k_setting_list}
            self.sparse_counts = {k: 0 for k in self.recall_k_setting_list}
            self.fusion_counts = {k: 0 for k in self.recall_k_setting_list}
        self.counter = {k: 0 for k in [0, 1, 2]}
        # k=0代表r@1候选集正确但是重排后错误
        # k=1代表r@1候选集错误，重排后错误但是候选集r@5包含正确答案
        # k=2代表r@1候选集错误，重排后错误且候选集r@5不包含正确答案
        self.max_statistical_error_counts = {k: 0 for k in [0, 3, 5, 8, 10]}
        self.min_statistical_error_counts = {k: 0 for k in [0, 3, 5, 8, 10]}
        self.statistical_error_counts = {k: 0 for k in [0, 3, 5, 8, 10]}
        if dataset.data_name == 'fashion-iq':
            self.dense_recall_lists = {dress_type: {k: [[None] for _ in range(dist.get_world_size())] for k in
                                       self.recall_k_setting_list} for dress_type in self.fashion_iq_list}
            self.sparse_recall_lists = {dress_type: {k: [[None] for _ in range(dist.get_world_size())] for k in
                                        self.recall_k_setting_list} for dress_type in self.fashion_iq_list}
            self.fusion_recall_lists = {dress_type: {k: [[None] for _ in range(dist.get_world_size())] for k in
                                        self.recall_k_setting_list} for dress_type in self.fashion_iq_list}
        else:
            self.dense_recall_lists = {k: [[None] for _ in range(dist.get_world_size())] for k in
                                       self.recall_k_setting_list}
            self.sparse_recall_lists = {k: [[None] for _ in range(dist.get_world_size())] for k in
                                        self.recall_k_setting_list}
            self.fusion_recall_lists = {k: [[None] for _ in range(dist.get_world_size())] for k in
                                        self.recall_k_setting_list}
        self.counter_lists = {k: [[None] for _ in range(dist.get_world_size())] for k in [0, 1, 2]}

        self.max_statistical_error_lists = {k: [[None] for _ in range(dist.get_world_size())] for k in [0, 3, 5, 8, 10]}
        self.min_statistical_error_lists = {k: [[None] for _ in range(dist.get_world_size())] for k in [0, 3, 5, 8, 10]}
        self.statistical_error_lists = {k: [[None] for _ in range(dist.get_world_size())] for k in [0, 3, 5, 8, 10]}

        self.dataset = dataset
        self.dense_run = dense_run
        self.sparse_run = sparse_run
        self.fusion_run = fusion_run

        self.look_up = look_up
        self.lookup_indices = lookup_indices
        self.search_args = search_args

        self.right_set = set()
        self.right_dict = {}
        self.wrong_set = set()
        self.wrong_dict = {}

    def _sort(self, dictionary):
        if self.dataset.data_name == 'coco' or self.dataset.data_name == 'flickr':
            sorted_by_value = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
            sorted_by_value_dicts = {k: dict(sorted_by_value[:k]) for k in self.recall_k_setting_list}
            search_results = {k: list(sorted_by_value_dicts[k]) for k in self.recall_k_setting_list}
            search_results = {k: torch.tensor([int(i) for i in search_results[k]]).cuda() for k in
                              self.recall_k_setting_list}
        elif self.dataset.data_name == 'fashion-iq':
            # 合成检索，key值是字符串，所以不能转换成张量
            sorted_by_value = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
            sorted_by_value_dicts = {k: dict(sorted_by_value[:k]) for k in self.recall_k_setting_list}
            search_results = {k: list(sorted_by_value_dicts[k]) for k in self.recall_k_setting_list}
        elif self.dataset.data_name == 'webqa':
            # 文本到图文检索，
            sorted_by_value = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
            sorted_by_value_dicts = {k: dict(sorted_by_value[:k]) for k in self.recall_k_setting_list}
            search_results = {k: list(sorted_by_value_dicts[k]) for k in self.recall_k_setting_list}
        else:
            # 行人检索，与图文检索并不一样，search_results应该再到数据集里面查询一下，根据img_id获取person_id
            sorted_by_value = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
            sorted_by_value_dicts = {k: dict(sorted_by_value[:k]) for k in self.recall_k_setting_list}
            search_results = {k: list(sorted_by_value_dicts[k]) for k in self.recall_k_setting_list}
            search_results = {k: torch.tensor([int(self.dataset.get_target_from_image(i)) for i in search_results[k]]).cuda() for k in
                              self.recall_k_setting_list}
        return search_results

    def _sort_return_id_and_value(self, dictionary):
        sorted_by_value = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
        sorted_by_value_dicts = {k: dict(sorted_by_value[:k]) for k in self.recall_k_setting_list}

        search_results = {k: list(sorted_by_value_dicts[k]) for k in self.recall_k_setting_list}
        search_scores = {k: list(sorted_by_value_dicts[k].values()) for k in self.recall_k_setting_list}
        search_results = {k: torch.tensor([int(i) for i in search_results[k]]).cuda() for k in
                          self.recall_k_setting_list}
        return search_results, search_scores

    def sort_and_count(self):
        if len(self.dense_run) > 0:
            for k, v in tqdm(self.dense_run.items()):
                if self.dataset.data_name == 'coco' or self.dataset.data_name == 'flickr':
                    target = self.dataset.get_target(k, self.search_args.query_type)
                    if isinstance(target, list):
                        target = torch.tensor([int(i) for i in target]).cuda()
                    else:
                        target = int(target)
                    if len(v['docs']) == 0:
                        continue

                    search_results = self._sort(v['docs'])
                    self._count('dense', search_results, target)
                elif self.dataset.data_name == 'fashion-iq':
                    target = self.dataset.get_target(k)
                    if len(v['docs']) == 0:
                        continue

                    search_results = self._sort(v['docs'])
                    self._count('dense', search_results, target)
                elif self.dataset.data_name == 'webqa':
                    target = self.dataset.get_target(k)
                    if len(v['docs']) == 0:
                        continue
                    search_results = self._sort(v['docs'])
                    self._count('dense', search_results, target)
                else:
                    target = self.dataset.get_target_from_text(k)
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
                if self.dataset.data_name == 'coco' or self.dataset.data_name == 'flickr':
                    target = self.dataset.get_target(k, self.search_args.query_type)
                    if isinstance(target, list):
                        target = torch.tensor([int(i) for i in target]).cuda()
                    else:
                        target = int(target)
                    if len(v['docs']) == 0:
                        continue

                    search_results = self._sort(v['docs'])
                    self._count('sparse', search_results, target)
                elif self.dataset.data_name == 'fashion-iq':
                    target = self.dataset.get_target(k)
                    if len(v['docs']) == 0:
                        continue

                    search_results = self._sort(v['docs'])
                    self._count('sparse', search_results, target)
                elif self.dataset.data_name == 'webqa':
                    target = self.dataset.get_target(k)
                    if len(v['docs']) == 0:
                        continue
                    search_results = self._sort(v['docs'])
                    self._count('dense', search_results, target)
                else:
                    target = self.dataset.get_target_from_text(k)
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
                if self.dataset.data_name == 'coco' or self.dataset.data_name == 'flickr':
                    target = self.dataset.get_target(k, self.search_args.query_type)
                    if isinstance(target, list):
                        target = torch.tensor([int(i) for i in target]).cuda()
                    else:
                        target = int(target)
                    if len(v) == 0:
                        continue

                    search_results = self._sort(v)
                    if True in torch.isin(search_results[1], target):
                        self.right_set.add(k)
                        self.right_dict[k] = search_results[1]
                    else:
                        self.wrong_set.add(k)
                        self.wrong_dict[k] = search_results[1]
                    self._count('fusion', search_results, target)
                elif self.dataset.data_name == 'fashion-iq':
                    target = self.dataset.get_target(k)
                    if len(v) == 0:
                        continue

                    search_results = self._sort(v)
                    self._count('fusion', search_results, target)
                elif self.dataset.data_name == 'webqa':
                    target = self.dataset.get_target(k)
                    if len(v) == 0:
                        continue

                    search_results = self._sort(v)
                    self._count('fusion', search_results, target)
                else:
                    target = self.dataset.get_target_from_text(k)
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
            if self.dataset.data_name == 'coco' or self.dataset.data_name == 'flickr':
                if True in torch.isin(search_results[k], target):
                    if result_type == 'dense':
                        self.dense_counts[k] += 1
                    elif result_type == 'sparse':
                        self.sparse_counts[k] += 1
                    else:
                        self.fusion_counts[k] += 1
            elif self.dataset.data_name == 'fashion-iq':
                # 合成检索
                if target in search_results[k]:
                    dress_type = self.dataset.get_dress_type(target)
                    if result_type == 'dense':
                        self.dense_counts[dress_type][k] += 1
                        # self.dense_counts[k] += 1
                    elif result_type == 'sparse':
                        self.sparse_counts[dress_type][k] += 1
                    else:
                        self.fusion_counts[dress_type][k] += 1
            elif self.dataset.data_name == 'webqa':
                if target in search_results[k]:
                    if result_type == 'dense':
                        self.dense_counts[k] += 1
                    elif result_type == 'sparse':
                        self.sparse_counts[k] += 1
                    else:
                        self.fusion_counts[k] += 1
            elif self.dataset.data_name == 'remuq':
                if target in search_results[k]:
                    if result_type == 'dense':
                        self.dense_counts[k] += 1
                    elif result_type == 'sparse':
                        self.sparse_counts[k] += 1
                    else:
                        self.fusion_counts[k] += 1
            else:
                # 行人检索
                if True in torch.isin(search_results[k], target):
                    if result_type == 'dense':
                        self.dense_counts[k] += 1
                    elif result_type == 'sparse':
                        self.sparse_counts[k] += 1
                    else:
                        self.fusion_counts[k] += 1

    def all_gather_object(self):
        if self.dataset.data_name == 'fashion-iq':
            self.dense_counts = {dress_type: {k: self.dense_counts[dress_type][k] / self.fashion_iq_length[dress_type] for k in
                                 self.recall_k_setting_list} for dress_type in self.fashion_iq_list}
            for dress_type in self.fashion_iq_list:
                for k in self.recall_k_setting_list:
                    dist.all_gather_object(object_list=self.dense_recall_lists[dress_type][k], obj=self.dense_counts[dress_type][k])

            self.sparse_counts = {dress_type: {k: self.sparse_counts[dress_type][k] / self.fashion_iq_length[dress_type] for k in
                                  self.recall_k_setting_list} for dress_type in self.fashion_iq_list}
            for dress_type in self.fashion_iq_list:
                for k in self.recall_k_setting_list:
                    dist.all_gather_object(object_list=self.sparse_recall_lists[dress_type][k], obj=self.sparse_counts[dress_type][k])

            self.fusion_counts = {dress_type: {k: self.fusion_counts[dress_type][k] / self.fashion_iq_length[dress_type] for k in
                                  self.recall_k_setting_list} for dress_type in self.fashion_iq_list}
            for dress_type in self.fashion_iq_list:
                for k in self.recall_k_setting_list:
                    dist.all_gather_object(object_list=self.fusion_recall_lists[dress_type][k], obj=self.fusion_counts[dress_type][k])
        else:
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

    def print_recall(self, output_path):
        if dist.get_rank() == 0:
            print(len(self.lookup_indices) * dist.get_world_size())
            xlsx_data = []
            if self.dataset.data_name != 'fashion-iq':
                xlsx_data.append([1, 5, 10, 100, 200, 300, 400])
                if len(self.dense_run) > 0:
                    print(len(self.look_up))
                    dense_recalls = {k: sum(self.dense_recall_lists[k]) for k in self.recall_k_setting_list}
                    for k in self.recall_k_setting_list:
                        print('Dense recall @ {}: {}'.format(k, self.dense_recall_lists[k]))
                    print(
                        'Dense reps recall: r@1 {}, r@5 {}, r@10 {}, r@100 {}, r@200 {}, r@300 {}, r@400 {}'.format(
                            dense_recalls[1],
                            dense_recalls[5],
                            dense_recalls[10],
                            dense_recalls[100],
                            dense_recalls[200],
                            dense_recalls[300],
                            dense_recalls[400]
                        ))
                    xlsx_data.append([dense_recalls[1],
                                      dense_recalls[5],
                                      dense_recalls[10],
                                      dense_recalls[100],
                                      dense_recalls[200],
                                      dense_recalls[300],
                                      dense_recalls[400]])

                if len(self.sparse_run) > 0:
                    sparse_recalls = {k: sum(self.sparse_recall_lists[k]) for k in self.recall_k_setting_list}
                    for k in self.recall_k_setting_list:
                        print('Sparse recall @ {}: {}'.format(k, self.sparse_recall_lists[k]))
                    print(
                        'Sparse reps recall: r@1 {}, r@5 {}, r@10 {}, r@100 {}, r@200 {}, r@300 {}, r@400 {}'.format(
                            sparse_recalls[1],
                            sparse_recalls[5],
                            sparse_recalls[10],
                            sparse_recalls[100],
                            sparse_recalls[200],
                            sparse_recalls[300],
                            sparse_recalls[400]))
                    xlsx_data.append([sparse_recalls[1],
                                      sparse_recalls[5],
                                      sparse_recalls[10],
                                      sparse_recalls[100],
                                      sparse_recalls[200],
                                      sparse_recalls[300],
                                      sparse_recalls[400]])
                if len(self.fusion_run) > 0:
                    fusion_recalls = {k: sum(self.fusion_recall_lists[k]) for k in self.recall_k_setting_list}
                    for k in self.recall_k_setting_list:
                        print('Fusion/Hybrid recall @ {}: {}'.format(k, self.fusion_recall_lists[k]))
                    print(
                        'Fusion/Hybrid reps recall: r@1 {}, r@5 {}, r@10 {}, r@100 {}, r@200 {}, r@300 {}, r@400 {}'.format(
                            fusion_recalls[1],
                            fusion_recalls[5],
                            fusion_recalls[10],
                            fusion_recalls[100],
                            fusion_recalls[200],
                            fusion_recalls[300],
                            fusion_recalls[400]))
                    xlsx_data.append([fusion_recalls[1],
                                      fusion_recalls[5],
                                      fusion_recalls[10],
                                      fusion_recalls[100],
                                      fusion_recalls[200],
                                      fusion_recalls[300],
                                      fusion_recalls[400]])
                df = pd.DataFrame(xlsx_data[1:], columns=xlsx_data[0])

                # 将DataFrame写入Excel文件，index=False表示不写入行索引
                df.to_excel(output_path, index=False)
            else:
                xlsx_data.append([1, 5, 10, 20, 30, 50, 100, 200, 300, 400])
                for dress_type in self.fashion_iq_list:
                    if len(self.dense_run) > 0:
                        print(len(self.look_up))
                        print(dress_type)
                        dense_recalls = {
                            dress_type: {k: sum(self.dense_recall_lists[dress_type][k]) for k in self.recall_k_setting_list}}
                        for k in self.recall_k_setting_list:
                            print('Dense recall @ {}: {}'.format(k, self.dense_recall_lists[dress_type][k]))
                        print(
                            'Dense reps recall: r@1 {}, r@5 {}, r@10 {}, r@20 {}, r@30 {}, r@50 {}, r@100 {}, r@200 {}, r@300 {}, r@400 {}'.format(
                                dense_recalls[dress_type][1],
                                dense_recalls[dress_type][5],
                                dense_recalls[dress_type][10],
                                dense_recalls[dress_type][20],
                                dense_recalls[dress_type][30],
                                dense_recalls[dress_type][50],
                                dense_recalls[dress_type][100],
                                dense_recalls[dress_type][200],
                                dense_recalls[dress_type][300],
                                dense_recalls[dress_type][400]
                            ))
                        xlsx_data.append([dense_recalls[dress_type][1],
                                          dense_recalls[dress_type][5],
                                          dense_recalls[dress_type][10],
                                          dense_recalls[dress_type][20],
                                          dense_recalls[dress_type][30],
                                          dense_recalls[dress_type][50],
                                          dense_recalls[dress_type][100],
                                          dense_recalls[dress_type][200],
                                          dense_recalls[dress_type][300],
                                          dense_recalls[dress_type][400]])

                    if len(self.sparse_run) > 0:
                        sparse_recalls = {dress_type: {k: sum(self.sparse_recall_lists[dress_type][k]) for k in self.recall_k_setting_list}}
                        for k in self.recall_k_setting_list:
                            print('Sparse recall @ {}: {}'.format(k, self.sparse_recall_lists[dress_type][k]))
                        print(
                            'Sparse reps recall: r@1 {}, r@5 {}, r@10 {}, r@20 {}, r@30 {}, r@50 {}, r@100 {}, r@200 {}, r@300 {}, r@400 {}'.format(
                                sparse_recalls[dress_type][1],
                                sparse_recalls[dress_type][5],
                                sparse_recalls[dress_type][10],
                                sparse_recalls[dress_type][20],
                                sparse_recalls[dress_type][30],
                                sparse_recalls[dress_type][50],
                                sparse_recalls[dress_type][100],
                                sparse_recalls[dress_type][200],
                                sparse_recalls[dress_type][300],
                                sparse_recalls[dress_type][400]))
                        xlsx_data.append([sparse_recalls[dress_type][1],
                                          sparse_recalls[dress_type][5],
                                          sparse_recalls[dress_type][10],
                                          sparse_recalls[dress_type][20],
                                          sparse_recalls[dress_type][30],
                                          sparse_recalls[dress_type][50],
                                          sparse_recalls[dress_type][100],
                                          sparse_recalls[dress_type][200],
                                          sparse_recalls[dress_type][300],
                                          sparse_recalls[dress_type][400]])
                    if len(self.fusion_run) > 0:
                        fusion_recalls = {dress_type: {k: sum(self.fusion_recall_lists[dress_type][k]) for k in self.recall_k_setting_list}}
                        for k in self.recall_k_setting_list:
                            print('Fusion/Hybrid recall @ {}: {}'.format(k, self.fusion_recall_lists[dress_type][k]))
                        print(
                            'Fusion/Hybrid reps recall: r@1 {}, r@5 {}, r@10 {}, r@20 {}, r@30 {}, r@50 {}, r@100 {}, r@200 {}, r@300 {}, r@400 {}'.format(
                                fusion_recalls[dress_type][1],
                                fusion_recalls[dress_type][5],
                                fusion_recalls[dress_type][10],
                                fusion_recalls[dress_type][20],
                                fusion_recalls[dress_type][30],
                                fusion_recalls[dress_type][50],
                                fusion_recalls[dress_type][100],
                                fusion_recalls[dress_type][200],
                                fusion_recalls[dress_type][300],
                                fusion_recalls[dress_type][400]))
                        xlsx_data.append([fusion_recalls[dress_type][1],
                                          fusion_recalls[dress_type][5],
                                          fusion_recalls[dress_type][10],
                                          fusion_recalls[dress_type][20],
                                          fusion_recalls[dress_type][30],
                                          fusion_recalls[dress_type][50],
                                          fusion_recalls[dress_type][100],
                                          fusion_recalls[dress_type][200],
                                          fusion_recalls[dress_type][300],
                                          fusion_recalls[dress_type][400]])
                df = pd.DataFrame(xlsx_data[1:], columns=xlsx_data[0])

                # 将DataFrame写入Excel文件，index=False表示不写入行索引
                df.to_excel(output_path, index=False)

    def statistical_error_data(self, processor, candidate_pool):
        for k, v in self.fusion_run.items():
            target = self.dataset.get_target(k, self.search_args.query_type)
            if isinstance(target, list):
                target = torch.tensor([int(i) for i in target]).cuda()
            else:
                target = int(target)
            if len(v) == 0:
                continue
            search_results = self._sort(v)
            candidate = self._sort(candidate_pool[k])
            if True not in torch.isin(search_results[1], target):
                error_text = self.dataset.text_dict[str(search_results[1].item())]
                error_length = len(processor(text=error_text, return_tensors="pt")['input_ids'].squeeze().tolist()[1:])
                max_length_right = 0
                min_length_right = 10000
                candidate_right = 0
                for result in search_results[5]:
                    if True in torch.isin(result, target):
                        text = self.dataset.text_dict[str(result.item())]
                        length_right = len(processor(text=text, return_tensors="pt")['input_ids'].squeeze().tolist()[1:])
                        if length_right >= max_length_right:
                            max_length_right = length_right
                        if length_right <= min_length_right:
                            min_length_right = length_right
                if True in torch.isin(candidate[1], target):
                    text = self.dataset.text_dict[str(candidate[1].item())]
                    candidate_right = len(processor(text=text, return_tensors="pt")['input_ids'].squeeze().tolist()[1:])
                    if dist.get_rank() == 0:
                        print(k)
                        print(self.dataset.text_dict[str(search_results[1].item())])
                        print(search_results[5])
                        print(self.dataset.text_dict[str(candidate[1].item())])
                        print(candidate[5])
                if True in torch.isin(candidate[5], target):
                    if dist.get_rank() == 0:
                        print(k)
                        print(self.dataset.text_dict[str(search_results[1].item())])
                        print(search_results[5])
                        print(self.dataset.text_dict[str(candidate[1].item())])
                        print(candidate[5])
                if max_length_right - error_length >= 10:
                    self.max_statistical_error_counts[10] += 1
                elif max_length_right - error_length >= 8:
                    self.max_statistical_error_counts[8] += 1
                elif max_length_right - error_length >= 5:
                    self.max_statistical_error_counts[5] += 1
                elif max_length_right - error_length >= 3:
                    self.max_statistical_error_counts[3] += 1
                else:
                    self.max_statistical_error_counts[0] += 1
                if candidate_right - error_length >= 10:
                    self.statistical_error_counts[10] += 1
                elif candidate_right - error_length >= 8:
                    self.statistical_error_counts[8] += 1
                elif candidate_right - error_length >= 5:
                    self.statistical_error_counts[5] += 1
                elif candidate_right - error_length >= 3:
                    self.statistical_error_counts[3] += 1
                else:
                    self.statistical_error_counts[0] += 1

            if True not in torch.isin(search_results[1], target) and True in torch.isin(candidate[1], target):
                self.counter[0] += 1
            elif True not in torch.isin(search_results[1], target) and True not in torch.isin(candidate[1], target) and True in torch.isin(candidate[5], target):
                self.counter[1] += 1
            elif True not in torch.isin(search_results[1], target) and True not in torch.isin(candidate[1], target) and True not in torch.isin(candidate[5], target):
                self.counter[2] += 1

        for k in [0, 3, 5, 8, 10]:
            dist.all_gather_object(object_list=self.statistical_error_lists[k], obj=self.statistical_error_counts[k])
            dist.all_gather_object(object_list=self.max_statistical_error_lists[k], obj=self.max_statistical_error_counts[k])
            dist.all_gather_object(object_list=self.min_statistical_error_lists[k], obj=self.min_statistical_error_counts[k])

        for k in [0, 1, 2]:
            dist.all_gather_object(object_list=self.counter_lists[k], obj=self.counter[k])

        if dist.get_rank() == 0:
            print(self.statistical_error_lists)
            print(self.max_statistical_error_lists)
            print(self.min_statistical_error_lists)
            for k in self.statistical_error_lists:
                self.statistical_error_lists[k] = sum(self.statistical_error_lists[k])
            for k in self.max_statistical_error_lists:
                self.max_statistical_error_lists[k] = sum(self.max_statistical_error_lists[k])
            for k in self.min_statistical_error_lists:
                self.min_statistical_error_lists[k] = sum(self.min_statistical_error_lists[k])
            print(self.statistical_error_lists)
            print(self.max_statistical_error_lists)
            print(self.min_statistical_error_lists)

            print(self.counter_lists)
            for k in self.counter_lists:
                self.counter_lists[k] = sum(self.counter_lists[k])
            print(self.counter_lists)