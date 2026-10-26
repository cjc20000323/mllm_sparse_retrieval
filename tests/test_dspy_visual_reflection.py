from __future__ import annotations

import base64
import importlib
import json
import os
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The entrypoint uses script-relative imports. Scope its search path so the
# nested src/src package cannot shadow src.generate_* in other test modules.
with patch.object(sys, 'path', [str(PROJECT_ROOT / 'src'), *sys.path]):
    entry = importlib.import_module('dspy_prompt_schema_generation_encode_and_search')
    dataset_module = importlib.import_module('dataset')


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status
        self.ok = status == 200
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


class VisualReflectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = entry.DspyReflectionArguments(
            dspy_reflection_mode='multimodal',
            dspy_feedback_failures_per_direction=2,
            dspy_reflection_image_max_edge=128,
        )
        self.config_patch = patch.object(entry, '_DSPY_REFLECTION_CONFIG', self.config)
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.cache_patch = patch.object(entry, '_DSPY_METRIC_CACHE', {})
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)
        self.dataset = self.make_dataset('flickr')

    def make_dataset(self, name):
        dataset = entry.CrossModalRetrievalDataset.__new__(entry.CrossModalRetrievalDataset)
        dataset.data_name = name
        dataset.data_path = str(self.root / 'generated' / 'data' / name)
        dataset.uses_generated_validation_data = True
        dataset.img_dict = {'1': '1.png', '2': '2.png', '3': '3.png'}
        dataset.img_id_list = list(dataset.img_dict)
        dataset.img2filepath = {key: 'val2014' for key in dataset.img_dict}
        dataset.text_dict = {
            't1': 'A person in red rides a bicycle.',
            't2': 'A person in red stands beside a bicycle.',
            't3': 'A dog runs through a field.',
        }
        dataset.img2text = {'1': ['t1'], '2': ['t2'], '3': ['t3']}
        dataset.text2img = {'t1': '1', 't2': '2', 't3': '3'}
        subdir = 'val2014' if name == 'coco' else 'flickr30k-images'
        image_dir = Path(dataset.data_path) / subdir
        image_dir.mkdir(parents=True)
        for image_id, color in [('1', 'red'), ('2', 'blue'), ('3', 'green')]:
            with Image.new('RGBA', (1000, 500), color) as image:
                image.save(image_dir / f'{image_id}.png')
        return dataset

    def diagnostics(self):
        text = entry.collect_retrieval_diagnostics(
            self.dataset, {'t1': {'docs': {'2': 1.0, '1': 0.5}}}, 'text')
        image = entry.collect_retrieval_diagnostics(
            self.dataset, {'1': {'docs': {'t2': 1.0, 't1': 0.5}}}, 'image')
        return [
            {'name': 'text-to-image', 'score': 2 / 3, 'diagnostics': text},
            {'name': 'image-to-text', 'score': 2 / 3, 'diagnostics': image},
        ]

    def cache_candidate(self, directions):
        aspects = ['people', 'clothes', 'scene']
        key = entry._dspy_aspect_key(aspects)
        entry._DSPY_METRIC_CACHE[key] = {
            'score': 2 / 3, 'feedback': 'R@1=0, R@5=100, R@10=100.',
            'direction_results': directions,
        }
        reflective = {'generate': [{
            'Inputs': {'dataset_name': 'flickr'},
            'Generated Outputs': {'aspects': str(aspects)},
            'Feedback': entry._DSPY_METRIC_CACHE[key]['feedback'],
        }]}
        return reflective

    def test_image_paths_use_generated_files_then_original_fallback(self):
        for name in ('flickr', 'coco'):
            with self.subTest(dataset=name):
                dataset = self.dataset if name == 'flickr' else self.make_dataset(name)
                generated = Path(entry._reflection_image_path(dataset, '1'))
                self.assertIn('generated', str(generated))
                relative = generated.relative_to(Path(dataset.data_path))
                original = self.root / 'original' / 'data' / name / relative
                original.parent.mkdir(parents=True)
                generated.replace(original)
                with patch.object(dataset_module, 'dataset_path_prefix', str(self.root / 'original')):
                    self.assertEqual(entry._reflection_image_path(dataset, '1'), str(original))
                    original.unlink()
                    with self.assertRaises(FileNotFoundError):
                        entry._reflection_image_path(dataset, '1')

    def test_client_sends_real_bounded_rgb_image_and_exact_served_model_id(self):
        response = FakeResponse({'choices': [{'message': {'content': 'A red image.'}}]})
        image_path = entry._reflection_image_path(self.dataset, '1')
        client = entry.LlavaReflectionClient('openai/llava-next-72b', self.config)
        with patch('requests.request', return_value=response) as request:
            self.assertEqual(client.complete('Inspect the mismatch.', image_path), 'A red image.')

        kwargs = request.call_args.kwargs
        self.assertEqual(request.call_args.args[:2],
                         ('POST', 'http://127.0.0.1:8000/v1/chat/completions'))
        self.assertEqual(kwargs['json']['model'], 'llava-next-72b')
        messages = kwargs['json']['messages']
        self.assertEqual(len(messages), 1)
        content = messages[0]['content']
        self.assertEqual(content[0]['text'], 'Inspect the mismatch.')
        self.assertEqual(len(content), 2)
        uri = content[1]['image_url']['url']
        self.assertTrue(uri.startswith('data:image/jpeg;base64,'))
        self.assertNotIn(image_path, json.dumps(messages))
        with Image.open(BytesIO(base64.b64decode(uri.split(',', 1)[1]))) as image:
            self.assertEqual(image.mode, 'RGB')
            self.assertEqual(image.size, (128, 64))

    def test_preflight_checks_model_and_makes_one_image_request(self):
        client = entry.LlavaReflectionClient('served-72b', self.config)
        responses = [
            FakeResponse({'data': [{'id': 'served-72b'}]}),
            FakeResponse({'choices': [{'message': {'content': 'A person.'}}]}),
        ]
        with patch('requests.request', side_effect=responses) as request:
            entry.preflight_dspy_reflection(client, self.dataset)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args_list[0].args[0], 'GET')
        self.assertEqual(request.call_args_list[1].kwargs['json']['max_tokens'], 64)

    def test_preflight_rejects_unserved_model(self):
        client = entry.LlavaReflectionClient('/unserved/local/path', self.config)
        with patch('requests.request', return_value=FakeResponse({'data': [{'id': 'alias'}]})):
            with self.assertRaisesRegex(RuntimeError, 'model ID'):
                entry.preflight_dspy_reflection(client, self.dataset)

    def test_proposer_serializes_pairs_and_uses_same_model_for_synthesis(self):
        reflective = self.cache_candidate(self.diagnostics())
        client = entry.LlavaReflectionClient('served-72b', self.config)
        proposer = entry.RetrievalMultimodalInstructionProposer(
            client, self.config, str(self.root / 'reflections.jsonl'))
        requests_seen = []

        def respond(method, url, **kwargs):
            payload = kwargs['json']
            requests_seen.append(payload)
            content = payload['messages'][0]['content']
            image_count = sum(item['type'] == 'image_url' for item in content)
            self.assertLessEqual(image_count, 1)
            result = ('The image differs in riding versus standing.' if image_count else
                      'Generate aspects for actions, human-object relations, and scene.')
            return FakeResponse({'choices': [{'message': {'content': result}}]})

        with patch('requests.request', side_effect=respond):
            proposal = proposer({'generate': 'Generate retrieval aspects.'}, reflective, ['generate'])
            # Repeated reflection from the same parent reuses only the image
            # diagnoses; a new Instruction is still requested each attempt.
            proposer({'generate': 'Generate retrieval aspects.'}, reflective, ['generate'])

        self.assertIn('human-object relations', proposal['generate'])
        self.assertEqual(len(requests_seen), 5)  # 3 images + 2 syntheses
        prompts = [payload['messages'][0]['content'][0]['text'] for payload in requests_seen]
        self.assertIn('wrong_top1_image', prompts[0])
        self.assertIn('correct_image', prompts[1])
        self.assertIn('query_image', prompts[2])
        self.assertIn('rides a bicycle', prompts[0])
        self.assertIn('stands beside', prompts[2])
        self.assertEqual({payload['model'] for payload in requests_seen}, {'served-72b'})
        log = (self.root / 'reflections.jsonl').read_text(encoding='utf-8')
        self.assertNotIn('base64', log)
        record = json.loads(log.splitlines()[0])
        self.assertEqual(len(record['cases']), 2)
        self.assertEqual(record['cases'][0]['analyses'][0]['image_role'], 'wrong_top1_image')

    def test_no_failures_reflects_on_metrics_without_images(self):
        directions = [{'name': 'image-to-text', 'score': 1.0, 'diagnostics': {
            'query_type': 'image', 'query_count': 1, 'recall': {1: 100, 5: 100, 10: 100},
            'failures': [],
        }}]
        reflective = self.cache_candidate(directions)
        client = SimpleNamespace(model_name='served', complete=lambda *args, **kwargs: 'New instruction')
        proposer = entry.RetrievalMultimodalInstructionProposer(
            client, self.config, str(self.root / 'empty.jsonl'))
        self.assertEqual(proposer({'generate': 'Old'}, reflective, ['generate']),
                         {'generate': 'New instruction'})

    def test_non_main_rank_never_calls_visual_service(self):
        client = SimpleNamespace(preflight=lambda *args: self.fail('non-main API call'))
        proposer = entry.RetrievalMultimodalInstructionProposer(client, self.config, 'unused')
        with (
            patch.object(entry, 'is_main_process', return_value=False),
            patch.object(entry, 'broadcast_object_from_main', return_value={'error': None}),
        ):
            entry.preflight_dspy_reflection(client, self.dataset)
            with self.assertRaisesRegex(RuntimeError, 'rank 0'):
                proposer({}, {}, [])

    def test_api_failure_does_not_record_a_completed_visual_proposal(self):
        reflective = self.cache_candidate(self.diagnostics())
        client = entry.LlavaReflectionClient('served', self.config)
        log_path = self.root / 'failed.jsonl'
        proposer = entry.RetrievalMultimodalInstructionProposer(client, self.config, str(log_path))
        completed_before = entry.get_completed_dspy_round_count()
        with patch('requests.request', return_value=FakeResponse({'error': 'vision failed'}, 500)):
            with self.assertRaisesRegex(RuntimeError, 'HTTP 500'):
                proposer({'generate': 'Old'}, reflective, ['generate'])
        self.assertFalse(log_path.exists())
        self.assertEqual(proposer.proposal_count, 0)
        self.assertEqual(entry.get_completed_dspy_round_count(), completed_before)

    def test_missing_selected_image_fails_before_visual_reflection(self):
        Path(entry._reflection_image_path(self.dataset, '2')).unlink()
        with patch.object(dataset_module, 'dataset_path_prefix', str(self.root / 'missing')):
            with self.assertRaisesRegex(RuntimeError, 'Reflection image'):
                entry.collect_retrieval_diagnostics(
                    self.dataset, {'t1': {'docs': {'2': 1.0, '1': 0.5}}}, 'text')

    def test_incomplete_instruction_is_not_accepted(self):
        client = entry.LlavaReflectionClient('served', self.config)
        payload = {'choices': [{
            'message': {'content': 'An unfinished Instruction'}, 'finish_reason': 'length',
        }]}
        with patch('requests.request', return_value=FakeResponse(payload)):
            with self.assertRaisesRegex(RuntimeError, 'Instruction was truncated'):
                client.complete('Rewrite the Instruction.')

    def test_api_error_does_not_echo_image_data_or_credentials(self):
        self.config.dspy_api_key = 'private-key'
        client = entry.LlavaReflectionClient('served', self.config)
        payload = {'error': 'private-key invalid image data:image/jpeg;base64,QUJDREVGRw=='}
        with patch('requests.request', return_value=FakeResponse(payload, 400)):
            with self.assertRaises(RuntimeError) as raised:
                client.complete('Inspect this image.')
        message = str(raised.exception)
        self.assertIn('HTTP 400', message)
        self.assertNotIn('QUJDREVGRw', message)
        self.assertNotIn('private-key', message)

    def test_diagnostics_bounds_gathered_failure_content_and_deduplicates_hits(self):
        sent = []
        sparse_run = {
            't1': {'docs': {'2': 1.0}},
            't2': {'docs': {'1': 1.0, '2': 0.5}},
            't3': {'docs': {'1': 1.0, '3': 0.5}},
        }

        def gather(local, gathered, dst):
            sent.append(local)
            gathered[:] = [local, local]  # DistributedSampler padding duplicate.

        with (
            patch.object(entry, 'is_distributed', return_value=True),
            patch.object(entry.dist, 'get_world_size', return_value=2),
            patch.object(entry.dist, 'get_rank', return_value=0),
            patch.object(entry.dist, 'gather_object', side_effect=gather),
            patch.object(entry, 'broadcast_object_from_main', side_effect=lambda value: value),
        ):
            result = entry.collect_retrieval_diagnostics(self.dataset, sparse_run, 'text')

        self.assertEqual(result['query_count'], 3)
        self.assertEqual(len(sent[0]['failures']), 2)
        self.assertNotIn('images', sent[0]['failures'][0])
        self.assertEqual(result['failures'][0]['query_id'], 't1')
        self.assertEqual(result['recall'][1], 0)
        self.assertAlmostEqual(result['recall'][5], 200 / 3)
        self.assertIn('images', result['failures'][0])

    def test_text_mode_remains_image_free_and_lock_is_isolated(self):
        visual_lock = entry.get_locked_dspy_aspects_path('flickr', 'light', 'served', str(self.root))
        with patch.object(entry, '_DSPY_REFLECTION_CONFIG', entry.DspyReflectionArguments()):
            text_lock = entry.get_locked_dspy_aspects_path('flickr', 'light', 'served', str(self.root))
            with patch.object(entry, '_reflection_image_path', side_effect=AssertionError('text mode image')):
                diagnostics = entry.collect_retrieval_diagnostics(
                    self.dataset, {'t1': {'docs': {'2': 1.0}}}, 'text')
            self.assertNotIn('images', diagnostics['failures'][0])
        self.assertNotEqual(visual_lock, text_lock)

    def test_config_cli_and_key_do_not_leak_into_metadata(self):
        parser = entry.HfArgumentParser(entry.DspyReflectionArguments)
        (config,) = parser.parse_args_into_dataclasses([
            '--dspy_reflection_mode', 'multimodal',
            '--dspy_feedback_failures_per_direction', '2',
        ])
        config.validate('coco', 'ir')
        self.assertEqual(config.dspy_feedback_failures_per_direction, 2)
        with patch.dict(os.environ, {'DSPY_API_KEY': 'private-key'}):
            self.assertEqual(config.api_key(), 'private-key')
            self.assertNotIn('private-key', json.dumps(config.identity()))
            self.assertNotIn('private-key', repr(config))
        with self.assertRaisesRegex(ValueError, 'Flickr/COCO'):
            config.validate('webqa', 't2it')


if __name__ == '__main__':
    unittest.main()
