from __future__ import annotations

import ast
import inspect
import io
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import reranker
from rerank_batching import (
    RerankRequest, configure_rerank_processor, indexed_batches, last_valid_logits,
    mean_target_nll, target_labels,
)
from time_estimation import StageTimeObserver
import test_freeret_time_estimation as fixtures
import test_rerank_time_estimation as timing_fixtures


class QuietProgress:
    def __init__(self, iterable=None, **kwargs):
        self.iterable = iterable
        self.n = 0

    def __iter__(self):
        return iter(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def update(self, value):
        self.n += value

    @staticmethod
    def write(value):
        pass


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        body = ''.join('<|vision_start|><|image_pad|><|vision_end|>' if part['type'] == 'image' else part['text']
                       for part in messages[0]['content'])
        return '<|im_start|>user\n' + body + '<|im_end|>\n<|im_start|>assistant\n'

    def __init__(self, *, bos=True, eos=True):
        self.tokenizer = SimpleNamespace(padding_side='right', pad_token_id=0, eos_token_id=2)
        self.bos, self.eos = bos, eos
        self.calls = []

    def encode(self, text):
        return ([1] if self.bos else []) + [3 + ord(char) % 251 for char in text] + ([2] if self.eos else [])

    def __call__(self, text, images=None, padding=False, return_tensors=None):
        assert padding is True
        assert return_tensors == 'pt'
        assert isinstance(text, list)
        self.calls.append({'texts': text, 'images': images})
        rows = [self.encode(value) for value in text]
        length = max(map(len, rows))
        ids, masks = [], []
        for row in rows:
            pads = [self.tokenizer.pad_token_id] * (length - len(row))
            if self.tokenizer.padding_side == 'left':
                ids.append(pads + row)
                masks.append([0] * len(pads) + [1] * len(row))
            else:
                ids.append(row + pads)
                masks.append([1] * len(row) + [0] * len(pads))
        if images:
            if isinstance(images[0], list):
                groups = images
            else:
                width = len(images) // len(text)
                assert width * len(text) == len(images)
                groups = [images[index * width:(index + 1) * width] for index in range(len(text))]
            keys = [sum((i + 1) * image.getpixel((0, 0))[0] for i, image in enumerate(group)) for group in groups]
        else:
            keys = [0] * len(text)
        return {'input_ids': torch.tensor(ids), 'attention_mask': torch.tensor(masks), 'image_keys': torch.tensor(keys)}


class OnevisionProcessor(Processor):
    pass


class Model:
    device = torch.device('cpu')

    def __init__(self, clock=None, vocab_size=256):
        self.calls = []
        self.clock, self.vocab_size = clock, vocab_size
        self.config = SimpleNamespace(vision_config=SimpleNamespace(patch_size=14), vision_feature_select_strategy='default')

    def __call__(self, input_ids, attention_mask, image_keys, *, use_cache, output_hidden_states, return_dict):
        assert not use_cache and not output_hidden_states and return_dict
        self.calls.append({'size': len(input_ids), 'images': image_keys.tolist(), 'mask': attention_mask.clone()})
        if self.clock is not None:
            self.clock.advance(1)
        code = (input_ids * attention_mask).cumsum(-1).float() % 37 + image_keys[:, None]
        vocabulary = torch.arange(self.vocab_size).float()
        logits = torch.sin((code[:, :, None] + vocabulary) / 13)
        return SimpleNamespace(logits=logits)


class BatchMathTests(unittest.TestCase):
    def test_grouping_restores_positions_and_preserves_tail(self):
        values = [0, 1, 1, 2, 0, 1]
        batches = list(indexed_batches(values, 4, key=lambda value: value))
        self.assertEqual([[index for index, _ in batch] for batch in batches], [[0, 4], [1, 2, 5], [3]])
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                list(indexed_batches(values, invalid))

    def test_last_logits_support_both_padding_sides(self):
        logits = torch.arange(2 * 4 * 3).view(2, 4, 3)
        mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]])
        torch.testing.assert_close(last_valid_logits(logits, mask), torch.stack([logits[0, 1], logits[1, 3]]))
        with self.assertRaises(ValueError):
            last_valid_logits(logits, torch.zeros(2, 4))
        with self.assertRaisesRegex(ValueError, 'sequence sizes'):
            last_valid_logits(logits, torch.ones(2, 3))

    def test_nll_labels_match_per_sample_for_padding_bos_and_eos_variants(self):
        for side in ('left', 'right'):
            for bos in (False, True):
                for eos in (False, True):
                    with self.subTest(side=side, bos=bos, eos=eos):
                        processor, model = Processor(bos=bos, eos=eos), Model()
                        processor.tokenizer.padding_side = side
                        prefixes = ['<image> prompt: ', 'longer prompt: ']
                        texts = [prefixes[0] + 'a', prefixes[1] + 'a longer answer']
                        inputs = processor(text=texts, padding=True, return_tensors='pt')
                        pre = processor(text=prefixes, padding=True, return_tensors='pt')
                        labels = target_labels(inputs['input_ids'], inputs['attention_mask'], pre['input_ids'], pre['attention_mask'])
                        output = model(**inputs, use_cache=False, output_hidden_states=False, return_dict=True)
                        actual = mean_target_nll(output.logits, labels, inputs['attention_mask'])
                        expected = []
                        for row, text in enumerate(texts):
                            single = processor(text=[text], padding=True, return_tensors='pt')
                            single_pre = processor(text=[prefixes[row]], padding=True, return_tensors='pt')
                            lab = target_labels(single['input_ids'], single['attention_mask'], single_pre['input_ids'], single_pre['attention_mask'])
                            out = model(**single, use_cache=False, output_hidden_states=False, return_dict=True)
                            mask = lab[:, 1:] != -100
                            expected.append(F.cross_entropy(out.logits[:, :-1][mask].float(), lab[:, 1:][mask]))
                        torch.testing.assert_close(actual, torch.stack(expected))
                        self.assertTrue(torch.all(labels[inputs['attention_mask'] == 0] == -100))

    def test_prefix_alignment_includes_boundary_merge_and_ignores_expanded_image_tokens(self):
        full = torch.tensor([[1, 200, 200, 8, 55, 4, 2, 0], [0, 1, 200, 200, 8, 56, 7, 2]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1]])
        prefix = torch.tensor([[1, 200, 200, 8, 9, 2], [1, 200, 200, 8, 9, 2]])
        labels = target_labels(full, mask, prefix, torch.ones_like(prefix))
        self.assertEqual(labels[0].tolist(), [-100, -100, -100, -100, 55, 4, 2, -100])
        self.assertEqual(labels[1].tolist(), [-100, -100, -100, -100, -100, 56, 7, 2])
        with self.assertRaisesRegex(ValueError, 'no scoreable'):
            target_labels(prefix, torch.ones_like(prefix), prefix, torch.ones_like(prefix))

    def test_configures_padding_and_llava_metadata(self):
        processor = Processor()
        processor.patch_size = processor.vision_feature_select_strategy = None
        processor.tokenizer.pad_token_id = None
        configure_rerank_processor(Model(), processor, 'before_pad')
        self.assertEqual(processor.patch_size, 14)
        self.assertEqual(processor.vision_feature_select_strategy, 'default')
        self.assertEqual(processor.tokenizer.pad_token_id, 2)
        configure_rerank_processor(Model(), processor, 'after_pad')
        self.assertEqual(processor.tokenizer.padding_side, 'left')
        model = Model()
        model.config.vision_config.model_type = 'clip_vision_model'
        processor.num_additional_image_tokens = 0
        configure_rerank_processor(model, processor, 'before_pad')
        self.assertEqual(processor.num_additional_image_tokens, 1)


class RerankerBatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.side = 'after_pad'
        for manager in (patch.object(reranker, 'path_prefix', str(self.root)),
                        patch.object(reranker, 'tqdm', QuietProgress), redirect_stdout(io.StringIO())):
            manager.__enter__()
            self.addCleanup(manager.__exit__, None, None, None)

    @staticmethod
    def image_bytes(value):
        stream = io.BytesIO()
        Image.new('RGB', (2, 2), (value, 0, 0)).save(stream, format='PNG')
        return {'bytes': stream.getvalue()}

    def ranker(self, task='ir', query_type='image', size=5, *, processor=None, clock=None):
        name = 'CUHK-PEDES' if task == 'tbpr' else 'fashion-iq' if task == 'cir' else 'flickr'
        image_dir = self.root / 'data' / name / ('imgs' if task == 'tbpr' else 'images/images' if task == 'cir' else 'flickr30k-images')
        image_dir.mkdir(parents=True, exist_ok=True)
        text_map = {'q0': 'query caption'}
        paths = {'q0': 'reference.png', '42': 'reference.png'}
        Image.new('RGB', (2, 2), (2, 0, 0)).save(image_dir / 'reference.png')
        candidates = {}
        for index in range(size):
            key = f'c{index}'
            text = 'caption ' + 'long ' * index + str(index)
            text_map[key] = text
            paths[key] = f'{key}.png'
            Image.new('RGB', (2, 2), (index + 3, 0, 0)).save(image_dir / paths[key])
            candidates[key] = {'text': text, 'image': self.image_bytes(index + 3)} if task == 't2it' else text
        dataset = SimpleNamespace(
            get_query=lambda _: {'text': 'query text', 'image': self.image_bytes(2)} if task == 'it2t' else 'query text',
            get_candidate=candidates.__getitem__, get_text=lambda _: 'make it blue', get_dress_type=lambda _: 'toptee',
        )
        model = Model(clock)
        ranker = reranker.Reranker(model, processor or Processor(), name, query_type, text_map, None, paths,
                                   {'Yes': 3, 'No': 4, 'A': 5, 'B': 6}, dataset=dataset)
        query = '42_0' if task == 'cir' else 'q0'
        fusion = {query: {key: float(size - index) for index, key in enumerate(candidates)}}
        return ranker, fusion

    def invoke(self, ranker, fusion, task, batch_size, *, caption=False, modify='no', model_name='llava-hf-llava-v1.6-mistral-7b-hf', template=None):
        data = SimpleNamespace(reps_loc=self.side)
        training = SimpleNamespace(fp16=False, task_type=task)
        model_args = SimpleNamespace(model_name_or_path=model_name)
        if caption:
            return ranker.caption_generation_rerank(fusion, 'pointwise', 100, data, training, model_args,
                                                   SimpleNamespace(modify_type=modify), rerank_batch_size=batch_size,
                                                   rerank_prompt_type=template or 'caption_generation')
        return ranker.rerank(fusion, 'pointwise', 100, data, training, model_args, rerank_batch_size=batch_size,
                             rerank_prompt_type=template or 'relevant')

    def assert_scores_equal(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for query in expected:
            self.assertEqual(list(expected[query]), list(actual[query]))
            for candidate in expected[query]:
                self.assertAlmostEqual(expected[query][candidate], actual[query][candidate], places=5)

    def test_pointwise_all_tasks_templates_padding_sides_and_batch_sizes(self):
        for task, query_type in [('ir', 'image'), ('ir', 'text'), ('tbpr', 'text'), ('t2it', 'text'), ('it2t', 'image'), ('cir', 'composed')]:
            for self.side in ('before_pad', 'after_pad'):
                for batch_size in (2, 4):
                    with self.subTest(task=task, query_type=query_type, side=self.side, batch=batch_size):
                        ranker, fusion = self.ranker(task, query_type)
                        expected = self.invoke(ranker, fusion, task, 1)
                        ranker.model.calls.clear()
                        actual = self.invoke(ranker, fusion, task, batch_size)
                        self.assert_scores_equal(expected, actual)
                        self.assertEqual([call['size'] for call in ranker.model.calls], [batch_size] * (5 // batch_size) + ([5 % batch_size] if 5 % batch_size else []))

    def test_freeret_ab_and_llava_qwen_template_variants(self):
        for model in ('llava-hf-llava-v1.6-mistral-7b-hf', 'llava-hf-llava-v1.6-vicuna-7b-hf', 'Qwen2.5-VL-7B-Instruct', 'Qwen3-VL-8B-Instruct'):
            for template in ('relevant', 'freeret'):
                with self.subTest(model=model, template=template):
                    ranker, fusion = self.ranker()
                    expected = self.invoke(ranker, fusion, 'ir', 1, model_name=model, template=template)
                    self.assert_scores_equal(expected, self.invoke(ranker, fusion, 'ir', 4, model_name=model, template=template))

    def test_caption_all_valid_tasks_padding_and_modify_single(self):
        for task, query_type in [('ir', 'image'), ('ir', 'text'), ('tbpr', 'text'), ('t2it', 'text'), ('it2t', 'image')]:
            for self.side in ('before_pad', 'after_pad'):
                for modify in ('no', 'modify_single'):
                    with self.subTest(task=task, query_type=query_type, side=self.side, modify=modify):
                        ranker, fusion = self.ranker(task, query_type)
                        expected = self.invoke(ranker, fusion, task, 1, caption=True, modify=modify)
                        ranker.model.calls.clear()
                        self.assert_scores_equal(expected, self.invoke(ranker, fusion, task, 4, caption=True, modify=modify))
                        corrected = modify == 'modify_single' and (task == 'it2t' or task == 'ir' and query_type == 'image')
                        self.assertEqual([call['size'] for call in ranker.model.calls], [4, 4, 1, 1] if corrected else [4, 1])

    def test_caption_other_model_templates(self):
        for model in ('llava-hf-llava-v1.6-vicuna-7b-hf', 'Qwen2.5-VL-7B-Instruct', 'Qwen3-VL-8B-Instruct'):
            with self.subTest(model=model):
                ranker, fusion = self.ranker()
                expected = self.invoke(ranker, fusion, 'ir', 1, caption=True, model_name=model, template='what_caption_generation')
                self.assert_scores_equal(expected, self.invoke(ranker, fusion, 'ir', 4, caption=True, model_name=model, template='what_caption_generation'))

    def test_cir_two_image_order_and_onevision_grouping(self):
        ranker, fusion = self.ranker('cir', 'composed')
        self.invoke(ranker, fusion, 'cir', 4)
        self.assertEqual(ranker.model.calls[0]['images'], [2 + 2 * value for value in range(3, 7)])
        ranker.model.calls.clear()
        self.invoke(ranker, fusion, 'cir', 4, template='origin_old_relevant')
        self.assertEqual(ranker.model.calls[0]['images'], [value + 4 for value in range(3, 7)])
        ranker, fusion = self.ranker(processor=OnevisionProcessor())
        self.invoke(ranker, fusion, 'ir', 4)
        self.assertIsInstance(ranker.processor.calls[0]['images'][0], list)

    def test_empty_candidates_and_no_cross_query_mixing(self):
        ranker, fusion = self.ranker(size=3)
        ranker.img_path_map['q1'] = ranker.img_path_map['q0']
        fusion['q1'] = dict(fusion['q0'])
        fusion['empty'] = {}
        result = self.invoke(ranker, fusion, 'ir', 4)
        self.assertEqual([call['size'] for call in ranker.model.calls], [3, 3])
        self.assertEqual(result['empty'], {})
        for batch in (0, -1):
            with self.assertRaises(ValueError):
                self.invoke(ranker, fusion, 'ir', batch)

    def test_twenty_candidates_need_five_model_calls_not_twenty(self):
        for caption in (False, True):
            ranker, fusion = self.ranker(size=20)
            self.invoke(ranker, fusion, 'ir', 4, caption=caption)
            self.assertEqual([call['size'] for call in ranker.model.calls], [4] * 5)

    def test_estimation_counts_complete_queries_not_candidate_batches(self):
        clock = timing_fixtures.Clock()
        ranker, one = self.ranker(size=20, clock=clock)
        fusion = {f'q{index}': dict(one['q0']) for index in range(40)}
        ranker.img_path_map.update({key: 'reference.png' for key in fusion})
        observer = StageTimeObserver('rerank', early_stop=True, collectives=timing_fixtures.SingleRank(), clock=clock)
        with torch.no_grad():
            result = ranker._batch_rerank(fusion, 'ir', '<image> <sent>', 4, 'relevant', SimpleNamespace(reps_loc='after_pad'),
                                         rerank_num=20, progress_observer=observer)
        self.assertEqual(len(result), 25)
        self.assertEqual(len(ranker.model.calls), 125)
        self.assertEqual(observer.report['status'], 'estimated')
        self.assertEqual(observer.report['estimated_total_seconds'], 200)

    def test_empty_shard_still_joins_estimation_collectives(self):
        group = timing_fixtures.ThreadRanks(2)
        rankers = [self.ranker(size=5)[0], self.ranker(size=5)[0]]
        outputs, errors = {}, []
        def worker(rank):
            try:
                observer = StageTimeObserver('rerank', early_stop=True, collectives=group.rank_transport(rank))
                fusion = {'q0': {f'c{index}': float(index) for index in range(5)}} if rank == 0 else {}
                with torch.no_grad():
                    result = rankers[rank]._batch_rerank(
                        fusion, 'ir', '<image> <sent>', 4, 'relevant', SimpleNamespace(reps_loc='after_pad'),
                        rerank_num=5, progress_observer=observer,
                    )
                outputs[rank] = result, observer.report
            except BaseException as error:
                errors.append(error)
                group.sync.abort()
        workers = [threading.Thread(target=worker, args=(rank,), daemon=True) for rank in range(2)]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join(10)
        self.assertFalse(any(thread.is_alive() for thread in workers), 'Empty shard did not join collectives')
        self.assertEqual(errors, [])
        self.assertEqual(outputs[1][0], {})
        self.assertEqual(outputs[0][1], outputs[1][1])
        self.assertEqual(outputs[0][1]['status'], 'completed')


class FreeRetBatchTests(unittest.TestCase):
    def test_scoring_uses_each_rows_last_valid_token_and_clears_hook(self):
        freeret = fixtures.freeret
        for side in ('left', 'right'):
            processor, model = Processor(), Model(vocab_size=2)
            processor.tokenizer.padding_side = side
            wrapper = freeret.FreeRetQwen.__new__(freeret.FreeRetQwen)
            wrapper.model = model
            wrapper._attn_hidden = torch.ones(3)
            wrapper._prepare_inputs = lambda texts, paths: processor(text=texts, padding=True, return_tensors='pt')
            expected = torch.cat([wrapper.rerank([text], [[]]) for text in ['a', 'longer text', 'x']])
            actual = wrapper.rerank(['a', 'longer text', 'x'], [[], [], []])
            torch.testing.assert_close(actual, expected)
            self.assertIsNone(wrapper._attn_hidden)
            self.assertEqual(model.calls[-1]['size'], 3)

    def test_free_ret_grouping_restores_all_candidate_scores(self):
        freeret = fixtures.freeret
        candidates = [SimpleNamespace(candidate_id=index) for index in range(6)]
        counts = [0, 1, 1, 2, 0, 1]
        calls = []
        def score(texts, paths):
            calls.append(len(texts))
            self.assertEqual(len({len(value) for value in paths}), 1)
            return torch.tensor([float(text) for text in texts])
        with patch.object(freeret, 'rerank_prompt', side_effect=lambda key, query, candidate: (str(candidate.candidate_id), ['image'] * counts[candidate.candidate_id])):
            result = freeret.rerank_candidates(SimpleNamespace(rerank=score), SimpleNamespace(prompt_key='test', candidates=candidates), None, list(range(6)), 4)
        self.assertEqual(result, list(range(6)))
        self.assertEqual(calls, [2, 3, 1])

    def test_free_ret_twenty_candidates_use_five_batches(self):
        freeret = fixtures.freeret
        calls = []
        def score(texts, paths):
            calls.append(len(texts))
            return torch.ones(len(texts))
        with patch.object(freeret, 'rerank_prompt', return_value=('prompt', ['image'])):
            result = freeret.rerank_candidates(SimpleNamespace(rerank=score), SimpleNamespace(prompt_key='test', candidates=[None] * 20), None, list(range(20)), 4)
        self.assertEqual(len(result), 20)
        self.assertEqual(calls, [4] * 5)

    def test_free_ret_scoring_failure_releases_hook_and_keeps_error(self):
        wrapper = fixtures.freeret.FreeRetQwen.__new__(fixtures.freeret.FreeRetQwen)
        wrapper._attn_hidden = torch.ones(3)
        wrapper._prepare_inputs = lambda *_: {}
        def fail(**kwargs):
            raise RuntimeError('original model failure')
        wrapper.model = fail
        with self.assertRaisesRegex(RuntimeError, 'original model failure'):
            wrapper.rerank(['text'], [[]])
        self.assertIsNone(wrapper._attn_hidden)


class NativeProcessorBatchTests(unittest.TestCase):
    @staticmethod
    def tokenizer(tokenizer_class=None):
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        from transformers import PreTrainedTokenizerFast
        specials = ['<unk>', '<pad>', '<bos>', '<eos>', '<image>', '<|vision_start|>', '<|image_pad|>', '<|vision_end|>']
        vocabulary = {token: index for index, token in enumerate(specials + ['prompt', 'short', 'longer', 'answer', 'target'])}
        backend = Tokenizer(models.WordLevel(vocabulary, unk_token='<unk>'))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        backend.post_processor = processors.TemplateProcessing(
            single='<bos> $A <eos>', special_tokens=[('<bos>', 2), ('<eos>', 3)],
        )
        tokenizer_class = tokenizer_class or PreTrainedTokenizerFast
        return tokenizer_class(tokenizer_object=backend, unk_token='<unk>', pad_token='<pad>',
                               bos_token='<bos>', eos_token='<eos>', additional_special_tokens=specials[4:])

    def check_native_batch(self, processor, image_token, image_marker):
        ranker = reranker.Reranker.__new__(reranker.Reranker)
        ranker.processor = processor
        images = [[Image.new('RGB', (28, 28), (4, 0, 0)), Image.new('RGB', (56, 28), (7, 0, 0))],
                  [Image.new('RGB', (28, 56), (9, 0, 0)), Image.new('RGB', (56, 56), (11, 0, 0))]]
        prefixes = [f'prompt {image_marker} {image_marker} ', f'prompt longer {image_marker} {image_marker} ']
        texts = [prefixes[0] + 'short', prefixes[1] + 'longer answer target']
        for side in ('left', 'right'):
            with self.subTest(processor=type(processor).__name__, side=side):
                processor.tokenizer.padding_side = side
                batch = ranker._processor_batch(texts, images)
                prefix_batch = ranker._processor_batch(prefixes, images)
                labels = target_labels(batch['input_ids'], batch['attention_mask'], prefix_batch['input_ids'], prefix_batch['attention_mask'])
                self.assertEqual(batch['input_ids'].shape[0], 2)
                self.assertTrue((labels[batch['input_ids'] == image_token] == -100).all())
                for row in range(2):
                    single = ranker._processor_batch([texts[row]], [images[row]])
                    torch.testing.assert_close(batch['input_ids'][row, batch['attention_mask'][row].bool()],
                                               single['input_ids'][0, single['attention_mask'][0].bool()])
                self.assertTrue((labels[batch['attention_mask'] == 0] == -100).all())
                self.assertTrue(((labels != -100).sum(-1) > 0).all())
        return batch

    def test_native_llava_next_two_image_batches_expand_in_correct_order(self):
        from transformers import LlavaNextImageProcessor, LlavaNextProcessor
        processor = LlavaNextProcessor(
            image_processor=LlavaNextImageProcessor(size={'shortest_edge': 24}, crop_size={'height': 24, 'width': 24},
                                                    image_grid_pinpoints=[[24, 24]]),
            tokenizer=self.tokenizer(), patch_size=8, num_additional_image_tokens=1, vision_feature_select_strategy='default',
        )
        inputs = self.check_native_batch(processor, 4, '<image>')
        self.assertEqual(inputs['image_sizes'].tolist(), [[28, 28], [28, 56], [56, 28], [56, 56]])

    def test_native_qwen_two_image_batches_keep_grid_and_token_alignment(self):
        from transformers import Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen2_5_VLProcessor
        kwargs = dict(image_processor=Qwen2VLImageProcessor(min_pixels=28 * 28, max_pixels=56 * 56),
                      tokenizer=self.tokenizer(Qwen2TokenizerFast))
        if 'video_processor' in inspect.signature(Qwen2_5_VLProcessor.__init__).parameters:
            from transformers.video_processing_utils import BaseVideoProcessor
            # Only images are exercised. New SDKs require a video processor
            # object at construction even when no video is supplied.
            kwargs['video_processor'] = BaseVideoProcessor()
        processor = Qwen2_5_VLProcessor(**kwargs)
        inputs = self.check_native_batch(processor, 6, '<|vision_start|><|image_pad|><|vision_end|>')
        self.assertEqual(inputs['image_grid_thw'].shape[0], 4)


class BatchEntryTests(unittest.TestCase):
    def test_legacy_dispatch_and_time_report_use_selected_batch_size(self):
        source = (SRC / 'rerank.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'rerank_kwargs' for target in node.targets))
        namespace = {'search_args': SimpleNamespace(rerank_batch_size=4)}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<batch-dispatch>', 'exec'), namespace)
        self.assertEqual(namespace['rerank_kwargs']['rerank_batch_size'], 4)
        self.assertIn('"rerank_batch_size": search_args.rerank_batch_size', source)
        for filename in ('rerank.sh', 'rerank_generated_validation.sh'):
            script = (SRC.parent / 'scripts' / filename).read_text()
            self.assertIn('RERANK_BATCH_SIZE:-1', script)
            self.assertIn('--rerank_batch_size "${RERANK_BATCH_SIZE}"', script)

    def test_pipeline_keeps_forwarding_batch_size_in_both_scoring_modes(self):
        tree = ast.parse((SRC / 'three_stage_retrieval_pipeline.py').read_text(encoding='utf-8'))
        method = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_dispatch_stage3_rerank')
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        namespace = {'SimpleNamespace': SimpleNamespace}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])), '<pipeline-batch>', 'exec'), namespace)
        for template in ('relevant', 'caption_generation'):
            calls = []
            def score(*args, **kwargs):
                calls.append(kwargs)
                return {}
            ranker = SimpleNamespace(rerank=score, caption_generation_rerank=score)
            args = SimpleNamespace(rerank_batch_size=4, rerank_template=template, rerank_type='pointwise', rerank_num=20, modify_type='no')
            namespace['_dispatch_stage3_rerank'](ranker, {}, args, None, None, None)
            self.assertEqual(calls[0]['rerank_batch_size'], 4)


if __name__ == '__main__':
    unittest.main()
