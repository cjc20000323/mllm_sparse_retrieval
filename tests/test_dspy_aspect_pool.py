from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from itertools import combinations
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
with patch.object(sys, 'path', [str(ROOT / 'src'), *sys.path]):
    from dspy_aspect_pool import DspyAspectPool, DspyAspectSelectionArguments
    import dspy_prompt_schema_generation_encode_and_search as entry


class PoolDefinitionTests(unittest.TestCase):
    def test_normalization_and_initial_selection_use_file_order(self):
        pool = DspyAspectPool([' Object ', 'ACTION', 'Scene', 'color'], ['color', 'object', 'scene'])
        self.assertEqual(pool.aspects, ('object', 'action', 'scene', 'color'))
        self.assertEqual(pool.initial_aspects, ('object', 'scene', 'color'))
        self.assertEqual(pool.validate(['SCENE', ' color ', 'OBJECT']), ['object', 'scene', 'color'])
        self.assertEqual(pool.capacity, 5)
        self.assertEqual(DspyAspectPool(['one', 'two', 'three']).capacity, 1)
        self.assertEqual(len(DspyAspectPool(tuple(str(i) for i in range(8))).initial_aspects), 5)

    def test_invalid_pools_and_seeds_are_rejected(self):
        for values in (None, 'object', [], ['object', 'scene'], ['a', 'A', 'b'], ['a', '', 'c'], ['a', 1, 'c']):
            with self.subTest(values=values), self.assertRaises(ValueError):
                DspyAspectPool(values)
        for initial in ([], ['object'], ['object', 'action', 'other'], ['object', 'Object', 'scene']):
            with self.subTest(initial=initial), self.assertRaises(ValueError):
                DspyAspectPool(['object', 'action', 'scene', 'color'], initial)

    def test_raw_outputs_are_not_silently_truncated_or_deduplicated(self):
        pool = DspyAspectPool(tuple('abcdefg'))
        for candidate in ('abc', None, ['a', 'b'], list('abcdefg'), ['a', 'b', 'outside'], ['a', 'A', 'b', 'c'], ['a', 'b', 1]):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                pool.validate(candidate)

    def test_fallback_visits_each_combination_once_and_never_changes_words(self):
        pool = DspyAspectPool(tuple('abcd'))
        used = set()
        for _ in range(pool.capacity):
            candidate = pool.next_unused(used)
            self.assertEqual(pool.validate(candidate), candidate)
            used.add(tuple(reversed(candidate)))  # permutations still exclude the same set
        self.assertEqual(len(used), 5)
        with self.assertRaisesRegex(ValueError, 'used'):
            pool.next_unused(used)

    def test_json_configuration_and_content_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pool.json'
            path.write_text(json.dumps({'aspects': ['A', 'b', 'c', 'd']}), encoding='utf-8')
            pool = DspyAspectPool.from_file(path)
            self.assertEqual(pool.identity(), DspyAspectPool([' a ', 'B', 'c', 'd']).identity())
            self.assertNotEqual(pool.identity(), DspyAspectPool(['a', 'b', 'c', 'other']).identity())
            self.assertNotEqual(pool.identity(), DspyAspectPool(['a', 'b', 'c', 'd'], ['a', 'b', 'c']).identity())
            for payload in ([], {}, {'aspects': 'abc'}, {'aspects': ['a', 'b', 'c'], 'initial_aspects': None}):
                path.write_text(json.dumps(payload), encoding='utf-8')
                with self.assertRaises(ValueError):
                    DspyAspectPool.from_file(path)
            self.assertIsNone(DspyAspectSelectionArguments('free', '/missing').load_pool())
            with self.assertRaisesRegex(ValueError, 'requires'):
                DspyAspectSelectionArguments('pool').load_pool()


class FakeGenerator:
    def __init__(self, outputs, instruction='seed'):
        self.outputs = list(outputs)
        self.signature = SimpleNamespace(instructions=instruction)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        result = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(result, Exception):
            raise result
        return entry.dspy.Prediction(aspects=result)


class RestrictedOptimizationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='dspy pool ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pool = DspyAspectPool(['object', 'action', 'scene', 'color'])
        self.state = patch.multiple(entry, _DSPY_ASPECT_POOL=self.pool, _DSPY_ASPECT_SOURCES={},
            _DSPY_TRIAL_RECORDING=False, _DSPY_TRIAL_RESULTS={}, _DSPY_METRIC_CACHE={},
            _DSPY_TRIAL_EVALUATION_COUNT=0, _DSPY_TOTAL_ROUNDS=None, _DSPY_REQUESTED_ROUNDS=None,
            _DSPY_ROUND_LOG_PATH=None, _DSPY_INITIAL_TRIAL_ASPECTS=None,
            _DSPY_INITIAL_TRIAL_KEY=None, _DSPY_INITIAL_TRIAL_PENDING=False,
            _DSPY_SEEN_ASPECT_KEYS=set(), _DSPY_RESERVED_ASPECT_KEYS=set(),
            _DSPY_GENERATION_COUNTER=0, _DSPY_NOVELTY_FALLBACK_COUNT=0,
            _DSPY_INSTRUCTION_ASPECT_CACHE={}, _DSPY_ASPECT_INSTRUCTION_MAP={},
            _DSPY_REFLECTION_CONFIG=entry.DspyReflectionArguments())
        self.state.start()
        self.addCleanup(self.state.stop)
        self.addCleanup(entry.end_dspy_trial_recording)
        output = redirect_stdout(StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def begin(self, initial=None, rounds=10):
        entry.begin_dspy_trial_recording(initial, rounds, str(self.root / 'rounds.jsonl'))

    def program(self, outputs):
        program = entry.SchemaAspectProgram()
        program.generate = FakeGenerator(outputs)
        return program

    def forward(self, program):
        return program('flickr', 'ir', 'example caption')

    def test_cli_parser_loads_the_selected_json_pool(self):
        path = self.root / 'pool.json'
        path.write_text(json.dumps({'aspects': list(self.pool.aspects)}), encoding='utf-8')
        parser = entry.HfArgumentParser(DspyAspectSelectionArguments)
        (args,) = parser.parse_args_into_dataclasses([
            '--dspy_aspect_mode', 'pool', '--dspy_aspect_pool_file', str(path)])
        self.assertEqual(args.load_pool(), self.pool)

    def test_real_gepa_with_dummy_lm_stops_at_pool_capacity_including_one_candidate(self):
        from dspy.utils.dummies import DummyLM
        for words in (['object', 'action', 'scene', 'color'], ['object', 'action', 'scene']):
            pool = DspyAspectPool(words)
            with self.subTest(words=words), patch.object(entry, '_DSPY_ASPECT_POOL', pool):
                initial = list(pool.initial_aspects)
                entry.begin_dspy_trial_recording(initial, pool.capacity, str(self.root / 'actual_gepa.jsonl'), requested_rounds=10)
                stopper = entry.MaxGEPAOptimizationRounds(pool.capacity, entry.get_completed_dspy_round_count)
                self.addCleanup(stopper.close)
                # DummyLM formats untyped output fields; supply JSON text as a real LM would.
                lm = DummyLM([{'aspects': json.dumps(initial)}] * 100)
                program = entry.SchemaAspectProgram()
                program.generate.signature = program.generate.signature.with_instructions(
                    entry.build_gepa_seed_instruction('flickr', 'ir', initial))
                example = entry.dspy.Example(dataset_name='flickr', task_type='ir', seed_texts='caption',
                                             novelty_context='', aspects=initial).with_inputs(
                                                 'dataset_name', 'task_type', 'seed_texts', 'novelty_context')
                program.generate.demos = [example]
                def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
                    aspects = pool.validate(pred.aspects)
                    key = entry._metric_cache_key(aspects)
                    if key not in entry._DSPY_METRIC_CACHE:
                        score = 2.0 if 'color' in aspects and len(aspects) == 3 else 0.2
                        entry.record_dspy_trial_result(aspects, score)
                        entry._DSPY_METRIC_CACHE[key] = {'score': score, 'feedback': 'Choose an unused subset.',
                                                        'aspect_selection': pool.identity()}
                    return entry.dspy.Prediction(**entry._DSPY_METRIC_CACHE[key])
                def proposer(candidate, reflective_dataset, components_to_update):
                    return {name: candidate[name] + '\nChoose another allowed subset.' for name in components_to_update}
                optimizer = entry.SafeGEPA(metric=metric, auto=None, max_metric_calls=1 + 4 * (pool.capacity - 1),
                    reflection_minibatch_size=1, candidate_selection_strategy='current_best', reflection_lm=lm,
                    instruction_proposer=proposer, skip_perfect_score=False, use_merge=False, num_threads=1,
                    failure_score=0.0, perfect_score=100.0, log_dir=str(self.root / ('gepa_' + str(len(words)))),
                    track_stats=True, seed=0, gepa_kwargs={'stop_callbacks': stopper})
                with entry.dspy.context(lm=lm, adapter=entry.dspy.ChatAdapter()):
                    optimizer.compile(program, trainset=[example], valset=[example])
                self.assertEqual(entry.get_completed_dspy_round_count(), pool.capacity)
                self.assertEqual(stopper.completed_rounds, pool.capacity)
                self.assertEqual(len(entry.get_dspy_trial_results()), pool.capacity)

    def test_illegal_raw_generation_retries_then_accepts_an_exact_subset(self):
        self.begin()
        program = self.program([['object', 'scene', 'outside'], ['COLOR', 'scene', 'object']])
        result = self.forward(program)
        self.assertEqual(result.aspects, ['object', 'scene', 'color'])
        self.assertEqual(len(program.generate.calls), 2)
        self.assertIn('Mandatory closed aspect pool', program.generate.calls[0]['novelty_context'])
        self.assertEqual(entry._DSPY_ASPECT_SOURCES[entry._dspy_aspect_key(result.aspects)], 'model')
        self.assertFalse(entry._DSPY_METRIC_CACHE)  # no evaluation fabricated by generation

    def test_repeated_outputs_use_bounded_pool_fallback_and_stable_cache(self):
        self.begin()
        program = self.program([['object', 'action', 'scene']])
        first = self.forward(program).aspects
        program.generate.signature.instructions = 'new instruction'
        second = self.forward(program).aspects
        self.assertNotEqual(set(first), set(second))
        self.assertEqual(len(program.generate.calls), 1 + entry._DSPY_NOVELTY_MAX_RETRIES + 1)
        self.assertEqual(entry._DSPY_ASPECT_SOURCES[entry._dspy_aspect_key(second)], 'pool_fallback')
        self.assertEqual(self.forward(program).aspects, second)
        self.assertEqual(entry._DSPY_NOVELTY_FALLBACK_COUNT, 1)

    def test_cache_poisoning_is_rejected_before_predictor_or_retrieval(self):
        self.begin()
        program = self.program([['object', 'action', 'scene']])
        key = entry._dspy_instruction_key('flickr', 'ir', 'seed')
        entry._DSPY_INSTRUCTION_ASPECT_CACHE[key] = ['object', 'action', 'outside']
        with self.assertRaisesRegex(RuntimeError, 'outside'):
            self.forward(program)
        self.assertFalse(program.generate.calls)
        self.assertFalse(entry._DSPY_METRIC_CACHE)
        with redirect_stderr(StringIO()), self.assertRaises(ValueError):
            entry.dspy_metric(SimpleNamespace(), entry.dspy.Prediction(aspects=['object', 'action', 'outside']))
        self.assertFalse(entry._DSPY_METRIC_CACHE)

    def test_small_pool_finishes_all_legal_combinations_and_selects_earliest_best(self):
        self.begin(list(self.pool.initial_aspects))
        self.assertEqual(entry._DSPY_TOTAL_ROUNDS, 5)
        program = self.program([list(self.pool.initial_aspects)])
        stopper = entry.MaxGEPAOptimizationRounds(5, entry.get_completed_dspy_round_count)
        self.addCleanup(stopper.close)
        for index, score in enumerate((0.2, 0.9, 0.9, 0.4, 0.1)):
            program.generate.signature.instructions = f'candidate-{index}'
            aspects = self.forward(program).aspects
            entry.record_dspy_trial_result(aspects, score)
            entry._DSPY_METRIC_CACHE[entry._metric_cache_key(aspects)] = {
                'score': score, 'feedback': '', 'aspect_selection': self.pool.identity(),
            }
            self.assertEqual(stopper(SimpleNamespace(i=index)), index == 4)
        trials = entry.get_dspy_trial_results()
        self.assertEqual(len(trials), 5)
        self.assertEqual(trials[0]['evaluation_index'], 2)
        self.assertEqual(trials[0]['score'], 0.9)
        rows = [json.loads(line) for line in (self.root / 'rounds.jsonl').read_text().splitlines()]
        self.assertEqual(rows[0]['candidate_source'], 'initial')
        self.assertEqual(rows[-1]['candidate_source'], 'pool_fallback')
        self.assertTrue(all(row['requested_optimization_rounds'] == 10 and row['effective_optimization_rounds'] == 5 for row in rows))
        self.assertEqual({frozenset(row['aspects']) for row in rows},
                         {frozenset(combination) for size in (3, 4) for combination in combinations(self.pool.aspects, size)})

    def test_first_round_uses_chosen_aspects_in_actual_prompts_not_old_five_templates(self):
        action = entry.RetrievalAction(
            SimpleNamespace(task_type='ir'), SimpleNamespace(dataset_name='flickr', use_generated_validation_data=False),
            SimpleNamespace(), SimpleNamespace(), SimpleNamespace(dspy_strength='light'), None, None, {})
        initial = ['color', 'object', 'scene']
        action.set_initial_prompt_aspects(initial)
        text, image, fusion, uses_old = action._construct_prompt_lists(initial)
        self.assertFalse(uses_old)
        self.assertEqual(len(text), 3)
        self.assertEqual(text[0], 'Summary the object in above sentence in one word: ')
        self.assertEqual(image[-1], 'Summary the color in above image in one word: ')
        self.assertIn('scene', fusion[1])
        self.assertIn(entry._aspect_pool_suffix(), action.prompt_generation_args.dspy_strength)
        with self.assertRaises(ValueError):
            action._construct_prompt_lists(['outside', 'object', 'action'])

    def test_no_recording_path_is_still_constrained(self):
        program = self.program([['not', 'in', 'pool']])
        result = self.forward(program)
        self.assertEqual(self.pool.validate(result.aspects), result.aspects)
        self.assertEqual(len(program.generate.calls), entry._DSPY_NOVELTY_MAX_RETRIES + 1)

    def test_pool_locks_outputs_and_metric_caches_are_isolated(self):
        aspects = ['object', 'action', 'scene']
        path = entry.save_locked_dspy_aspects(aspects, 'flickr', 'light', 'model', str(self.root))
        self.assertEqual(entry.load_locked_dspy_aspects('flickr', 'light', 'model', str(self.root))[0], aspects)
        metric_key = entry._metric_cache_key(aspects)
        other = DspyAspectPool(['object', 'action', 'scene', 'material'])
        with patch.object(entry, '_DSPY_ASPECT_POOL', other):
            other_path = entry.get_locked_dspy_aspects_path('flickr', 'light', 'model', str(self.root))
            self.assertNotEqual(path, other_path)
            self.assertNotEqual(metric_key, entry._metric_cache_key(aspects))
        with patch.object(entry, '_DSPY_ASPECT_POOL', None):
            free_path = entry.get_locked_dspy_aspects_path('flickr', 'light', 'model', str(self.root))
            self.assertNotIn('_pool_', free_path)
        payload = json.loads(Path(path).read_text())
        payload['aspects'] = ['object', 'action', 'outside']
        Path(path).write_text(json.dumps(payload), encoding='utf-8')
        with self.assertRaises(ValueError):
            entry.load_locked_dspy_aspects('flickr', 'light', 'model', str(self.root))
        saved = entry.save_dspy_aspects(aspects, 'flickr', 'light', 'model', str(self.root))
        self.assertIn('_pool_', saved)
        self.assertEqual(json.loads(Path(saved).read_text())['aspect_selection'], self.pool.identity())

    def test_legal_metric_cache_hit_does_not_run_gpu_retrieval(self):
        aspects = ['object', 'action', 'scene']
        entry._DSPY_METRIC_CACHE[entry._metric_cache_key(aspects)] = {
            'score': 12.0, 'feedback': 'cached', 'aspect_selection': self.pool.identity(),
        }
        result = entry.dspy_metric(SimpleNamespace(), entry.dspy.Prediction(aspects=aspects))
        self.assertEqual(result.score, 12.0)

    def test_seed_proposal_and_textual_feedback_all_include_pool_rules(self):
        texts = [entry.build_gepa_seed_instruction('flickr', 'ir', self.pool.initial_aspects),
                 entry.make_unique_gepa_instruction_proposal({'generate': 'Rewrite freely'}, 1)['generate'],
                 entry.build_gepa_retrieval_feedback(self.pool.initial_aspects, 1.0, [])]
        self.assertTrue(all('Mandatory closed aspect pool' in text for text in texts))

    def test_visual_diagnosis_and_synthesis_share_the_constraint(self):
        aspects = ['object', 'action', 'scene']
        failure = {'query_id': '1', 'query': 'query', 'correct_target': 'correct caption',
                   'wrong_top1': 'wrong caption', 'target_rank': 2,
                   'images': [{'path': '/unused/image.jpg', 'role': 'query_image', 'image_id': '1'}]}
        entry._DSPY_METRIC_CACHE[entry._metric_cache_key(aspects)] = {
            'score': 0.5, 'feedback': 'failure', 'aspect_selection': self.pool.identity(),
            'direction_results': [{'name': 'image-to-text', 'diagnostics': {'failures': [failure]}}],
        }
        client = SimpleNamespace(model_name='fake', complete=Mock(return_value='Use the allowed subset.'))
        proposer = entry.RetrievalMultimodalInstructionProposer(client, entry.DspyReflectionArguments(), str(self.root / 'reflections.jsonl'))
        examples = {'generate': [{'Generated Outputs': {'aspects': aspects}}]}
        self.assertEqual(proposer({'generate': 'old'}, examples, ['generate']), {'generate': 'Use the allowed subset.'})
        self.assertEqual(client.complete.call_count, 2)
        self.assertTrue(all('Mandatory closed aspect pool' in call.args[0] for call in client.complete.call_args_list))
        metadata = json.loads((self.root / 'reflections.jsonl').read_text())
        self.assertEqual(metadata['aspect_selection'], self.pool.identity())

    def test_worker_uses_roots_choice_and_provenance(self):
        self.begin()
        program = self.program([['object', 'scene', 'color']])
        packets = []
        def root_broadcast(payload):
            packets.append(pickle.loads(pickle.dumps(payload)))
            return payload
        with patch.object(entry, 'broadcast_object_from_main', side_effect=root_broadcast):
            root_result = self.forward(program)
        worker = self.program([['object', 'action', 'scene']])
        with (
            patch.object(entry, 'is_main_process', return_value=False),
            patch.object(entry, 'broadcast_object_from_main', return_value=packets[-1]),
            patch.object(entry, '_DSPY_INSTRUCTION_ASPECT_CACHE', {}),
            patch.object(entry, '_DSPY_SEEN_ASPECT_KEYS', set()),
            patch.object(entry, '_DSPY_GENERATION_COUNTER', 0),
        ):
            result = self.forward(worker)
            self.assertEqual(result.aspects, root_result.aspects)
            self.assertEqual(entry._DSPY_GENERATION_COUNTER, 1)
            self.assertEqual(len(entry._DSPY_SEEN_ASPECT_KEYS), 1)

    def test_service_failure_is_not_replaced_by_a_fabricated_candidate(self):
        self.begin()
        program = self.program([RuntimeError('service unavailable')])
        with self.assertRaisesRegex(RuntimeError, 'service unavailable'):
            self.forward(program)
        self.assertFalse(entry._DSPY_SEEN_ASPECT_KEYS)
        self.assertFalse(entry._DSPY_METRIC_CACHE)

    def test_worker_error_is_propagated_before_committing_any_candidate(self):
        self.begin()
        program = self.program([['object', 'action', 'scene']])
        def gather(errors, local_error):
            errors[:] = [local_error, 'worker service failed']
        with (
            patch.object(entry, 'is_distributed', return_value=True),
            patch.object(entry, 'is_main_process', return_value=True),
            patch.object(entry.dist, 'get_world_size', return_value=2),
            patch.object(entry.dist, 'all_gather_object', side_effect=gather),
            patch.object(entry, 'broadcast_object_from_main', side_effect=lambda value: value),
        ):
            with self.assertRaisesRegex(RuntimeError, 'worker service failed'):
                self.forward(program)
        self.assertFalse(entry._DSPY_SEEN_ASPECT_KEYS)
        self.assertFalse(entry._DSPY_METRIC_CACHE)


BASH = shutil.which('bash') or next((str(path) for path in (Path('D:/Git/bin/bash.exe'), Path('C:/Program Files/Git/bin/bash.exe')) if path.is_file()), None)


@unittest.skipUnless(BASH, 'Bash required')
class PoolLauncherTests(unittest.TestCase):
    def test_both_scripts_forward_pool_configuration_and_preserve_free_default(self):
        with tempfile.TemporaryDirectory(prefix='dspy pool launcher ') as directory:
            root = Path(directory)
            (root / 'scripts').mkdir(); (root / 'bin').mkdir()
            (root / 'generated/data').mkdir(parents=True)
            pool = root / 'pool with spaces.json'
            pool.write_text(json.dumps({'aspects': ['object', 'action', 'scene']}), encoding='utf-8')
            recorder = root / 'bin/deepspeed'
            recorder.write_bytes(b'#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$RECORD"\n')
            recorder.chmod(0o755)
            record = root / 'args.bin'
            env = {**os.environ, 'PATH': str(root / 'bin') + os.pathsep + str(Path(BASH).parent) + os.pathsep + os.environ.get('PATH', ''),
                   'RECORD': record.as_posix(), 'GENERATED_VALIDATION_DATA_ROOT': (root / 'generated').as_posix(),
                   'DSPY_ASPECT_MODE': 'free', 'DSPY_ASPECT_POOL_FILE': ''}
            for suffix in ('', '_generated_validation'):
                name = 'dspy_prompt_schema_generation_encode_and_search' + suffix + '.sh'
                script = root / 'scripts' / name
                shutil.copyfile(ROOT / 'scripts' / name, script)
                syntax = subprocess.run([BASH, '-n', script.as_posix()], capture_output=True, timeout=30)
                self.assertEqual(syntax.returncode, 0, syntax.stderr)
                for mode in ('free', 'pool'):
                    result = subprocess.run([BASH, script.as_posix()], cwd=root.parent,
                        env={**env, 'DSPY_ASPECT_MODE': mode, 'DSPY_ASPECT_POOL_FILE': pool.as_posix() if mode == 'pool' else ''},
                        capture_output=True, encoding='utf-8', errors='replace', timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    args = record.read_bytes().decode().split('\0')[:-1]
                    self.assertEqual(args[args.index('--dspy_aspect_mode') + 1], mode)
                    if mode == 'pool':
                        self.assertEqual(args[args.index('--dspy_aspect_pool_file') + 1], pool.as_posix())
                    else:
                        self.assertNotIn('--dspy_aspect_pool_file', args)
                record.unlink()
                result = subprocess.run([BASH, script.as_posix()], env={**env, 'DSPY_ASPECT_MODE': 'pool'},
                    capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(record.exists())


if __name__ == '__main__':
    unittest.main()
