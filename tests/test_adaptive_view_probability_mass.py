"""Probability-prefix selection, immutable caches, exact lookup and variable-view encoding."""
from contextlib import redirect_stdout
import copy
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
import test_adaptive_view_shuffle as fixtures
from test_freeret_dense import ROOT, RealModelTests, data_args, model_args, temporary_directory
import adaptive_view_probability_mass as mass
import adaptive_views as views
import adaptive_view_encoding as encoding
import adaptive_encode_batch as encode_batch
import adaptive_retrieval_batch as retrieval
import adaptive_sparse_index_batch as indexing


def legacy_selection(source, threshold=0.7):
    """A v1 record as written before the stop-before-crossing correction."""
    result=copy.deepcopy(source)
    order=sorted(range(len(source['pool'])),key=lambda i:(-source['scores'][i],i))
    chosen,rows,stopped=[],[],False
    for rank,index in enumerate(order,1):
        cumulative=math.fsum(source['scores'][i] for i in order[:rank])
        if not stopped:
            chosen.append(dict(label=source['pool'][index]['label'],aspect=source['pool'][index]['aspect']))
        rows.append(dict(rank=rank,label=source['pool'][index]['label'],aspect=source['pool'][index]['aspect'],
            probability=source['scores'][index],cumulative_probability=cumulative,selected=not stopped))
        if threshold<1 and cumulative>threshold and not math.isclose(cumulative,threshold,rel_tol=0,abs_tol=1e-12):
            stopped=True
    result.update(version=mass.PMI_VERSION if views.is_pmi_selection(source) else mass.VERSION,
        selection_strategy=mass.STRATEGY,probability_source_fingerprint=source['fingerprint'],
        probability_mass_configuration=dict(strategy=mass.STRATEGY,version=1,threshold=float(threshold),comparison='gt',tolerance=1e-12),
        selected=chosen,probability_mass_ranking=rows,selected_count=len(chosen),
        selected_probability_mass=math.fsum(row['probability'] for row in rows if row['selected']))
    result['views_fingerprint']=mass.fingerprint(result)
    result['fingerprint']=views.digest({k:v for k,v in result.items() if k!='fingerprint'})
    return result


class SelectionTests(unittest.TestCase):
    def test_stop_before_crossing_equality_endpoints_and_stable_ties(self):
        pool = [dict(label=chr(65+i), aspect=str(i)) for i in range(4)]
        for threshold, labels in ((0, 'A'), (0.3, 'A'), (0.4, 'A'), (0.7, 'AB'), (0.9, 'ABC'), (1, 'ABCD')):
            selected, rows, total = mass.select(pool, [0.4, 0.3, 0.2, 0.1], threshold)
            self.assertEqual(''.join(row['label'] for row in selected), labels)
            self.assertEqual(sum(row['selected'] for row in rows), len(labels))
            self.assertAlmostEqual(total, math.fsum([0.4, 0.3, 0.2, 0.1][:len(labels)]))
        selected, _, _ = mass.select(pool, [0.25]*4, 0.5)
        self.assertEqual([row['label'] for row in selected], list('AB'))

    def test_rejected_view_stops_the_prefix_even_when_later_views_would_fit(self):
        pool=[dict(label=chr(65+i),aspect=str(i)) for i in range(4)]
        selected,rows,total=mass.select(pool,[0.4,0.3,0.2,0.1],0.65)
        self.assertEqual([row['label'] for row in selected],['A'])
        self.assertEqual([row['selected'] for row in rows],[True,False,False,False])
        self.assertEqual(total,0.4)

    def test_minimum_one_fallback_and_selection_budget(self):
        source=fixtures.choose(fixtures.options(shuffled=False))
        for threshold in (0,0.001,0.5,0.9,1):
            selected=mass.derive(source,threshold)
            total=selected['selected_probability_mass']
            if selected['minimum_one_fallback']:
                self.assertEqual(len(selected['selected']),1)
                self.assertGreater(total,threshold)
            else:
                self.assertLessEqual(total,threshold+mass.TOLERANCE)
            self.assertEqual(mass.validate_selection(selected),len(selected['selected']))
        self.assertTrue(mass.derive(source,0)['minimum_one_fallback'])
        self.assertFalse(mass.derive(source,1)['minimum_one_fallback'])

    def test_legacy_records_are_read_only_probability_sources_not_current_artifacts(self):
        with temporary_directory() as directory:
            path=Path(directory)/'legacy.json'
            for hierarchical in (False,True):
                for pmi in (False,True):
                    for shuffled in (False,True):
                        with self.subTest(hierarchy=hierarchical,pmi=pmi,shuffled=shuffled):
                            source=fixtures.choose(fixtures.options(shuffled=shuffled,pmi=pmi,hierarchical=hierarchical))
                            old=legacy_selection(source)
                            views.atomic_json(path,old)
                            before=path.read_bytes()
                            with self.assertRaisesRegex(views.IncompatibleViewSelectionError,'v1.*v2'):
                                views.load_selection(path)
                            with self.assertRaisesRegex(views.IncompatibleViewSelectionError,'v1.*v2'):
                                views.load_selection(path,view_probability_mass=True,allow_legacy_probability_mass_source=True)
                            loaded=views.load_selection(path,allow_legacy_probability_mass_source=True)
                            self.assertEqual(loaded,old)
                            corrected=mass.derive(loaded,0.7)
                            self.assertEqual(len(corrected['selected']),max(1,len(old['selected'])-1))
                            self.assertEqual(corrected[mass.FIELD]['version'],2)
                            self.assertEqual(corrected[mass.FIELD]['comparison'],'le')
                            self.assertEqual(mass.probability_source(corrected),source)
                            self.assertNotEqual(corrected['views_fingerprint'],old['views_fingerprint'])
                            self.assertEqual(path.read_bytes(),before)
                            # Reject legacy records before building or reusing an index.
                            model=model_args(False,'fixture')
                            data=data_args()
                            manifest=dict(encoding.metadata(model,data,'ir','image',fixtures.IDENTITY,old),
                                selection_file=str(path),selection_fingerprint=old['fingerprint'])
                            artifact=Path(directory)/'corpus'; views.atomic_json(artifact/views.MANIFEST,manifest)
                            with self.assertRaisesRegex(views.IncompatibleViewSelectionError,'v1.*v2'):
                                views.validate_index_selection(artifact,manifest)

    def test_invalid_thresholds_and_probability_vectors_fail(self):
        for value in (-0.1, 1.1, float('nan'), float('inf'), True, None, 'bad'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                mass.configuration(value)
        pool = [dict(label='A', aspect='a')]
        for scores in ([], [0], [float('nan')], [-1], [True], [1, 0]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                mass.select(pool, scores, 0.9)

    def test_derive_load_and_round_validation_for_pmi_shuffle_and_both_pools(self):
        with temporary_directory() as directory:
            path = Path(directory)/'selection.json'
            for hierarchical in (False, True):
                for pmi in (False, True):
                    for shuffled in (False, True):
                        for modality in ('text', 'image', 'both'):
                            with self.subTest(hierarchy=hierarchical, pmi=pmi, shuffle=shuffled, modality=modality):
                                source = fixtures.choose(fixtures.options(shuffled=shuffled, pmi=pmi,
                                    hierarchical=hierarchical, mode=modality))
                                before = copy.deepcopy(source)
                                selected = mass.derive(source, 0.75)
                                self.assertEqual(source, before)
                                self.assertEqual(mass.probability_source(selected), source)
                                self.assertEqual(selected['version'], mass.PMI_VERSION if pmi else mass.VERSION)
                                views.atomic_json(path, selected)
                                self.assertEqual(views.load_selection(path, view_probability_mass=True,
                                    view_probability_mass_threshold=0.75), selected)
                                for requested, threshold in ((False, None), (True, 0.8)):
                                    with self.assertRaisesRegex(ValueError, 'mode differs|threshold differs'):
                                        views.load_selection(path, view_probability_mass=requested,
                                            view_probability_mass_threshold=threshold)
                                # A new outer fingerprint cannot make forged decisions valid.
                                forged = copy.deepcopy(selected)
                                forged['probability_mass_ranking'][0]['cumulative_probability'] += 0.01
                                forged['fingerprint'] = views.digest({k:v for k,v in forged.items() if k!='fingerprint'})
                                views.atomic_json(path, forged)
                                with self.assertRaisesRegex(ValueError, 'probability_mass_ranking'):
                                    views.load_selection(path)

    def test_same_prefix_different_thresholds_have_separate_fingerprints(self):
        source = fixtures.choose(fixtures.options(shuffled=False))
        first, second = mass.derive(source, 0), mass.derive(source, 0.001)
        self.assertEqual(first['selected'], second['selected'])
        self.assertNotEqual(first['views_fingerprint'], second['views_fingerprint'])
        self.assertNotEqual(first['fingerprint'], second['fingerprint'])
        self.assertEqual(mass.derive(first, 0.9), mass.derive(source, 0.9))
        original = copy.deepcopy(source)
        self.assertEqual(views.views_fingerprint(source['selected'], source['selection_modality']), source['views_fingerprint'])
        self.assertEqual(source, original)

    def test_result_path_markers_are_idempotent_with_every_freeret_mode(self):
        import dense_representation as dense
        selected=mass.derive(fixtures.choose(fixtures.options(shuffled=False)),0.9)
        for use,prompts,full in ((False,False,False),(True,False,False),(True,True,False),(True,False,True)):
            path=dense.mode_path(Path('results')/'run',use,prompts,full)
            marked=mass.mark_output(path,selected)
            self.assertEqual(mass.mark_output(marked,selected),marked)
            self.assertEqual(dense.mode_path(marked,use,prompts,full),marked)
            self.assertIn('_pm_'+selected['views_fingerprint'][:16],marked.name)

    def test_bad_aggregate_and_selected_count_fail_even_with_new_outer_hash(self):
        selected = mass.derive(fixtures.choose(fixtures.options(shuffled=False)), 0.9)
        for key in ('scores', 'selected_count', 'probability_source_fingerprint'):
            forged = copy.deepcopy(selected)
            if key=='scores':
                forged[key][0], forged[key][-1] = forged[key][-1], forged[key][0]
            else:
                forged[key] = 999 if key=='selected_count' else '0'*64
            with self.subTest(field=key), self.assertRaises(ValueError):
                mass.validate_selection(forged)

    def test_overlap_rejected_without_altering_its_implementation(self):
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            mass.enabled(NS(view_probability_mass=True, view_overlap_selection=True))
        with self.assertRaisesRegex(ValueError, 'Overlap'):
            mass.derive(dict(version=views.OVERLAP_VERSION))


class CacheAndLookupTests(unittest.TestCase):
    def test_encoding_uses_explicit_probability_file_without_overwriting_it(self):
        import qwen_vl_pipeline
        with temporary_directory() as directory:
            root=Path(directory)
            options=fixtures.options(shuffled=False)
            options.view_probability_mass=True
            options.adaptive_data_root=str(root/'data')
            options.view_selection_file=str(root/'source.json')
            source=fixtures.choose(options,dataset='RSTPReid')
            source['selection_split']='train'
            source['fingerprint']=views.digest({k:v for k,v in source.items() if k!='fingerprint'})
            source=legacy_selection(source,0.8)
            views.atomic_json(options.view_selection_file,source)
            before=Path(options.view_selection_file).read_bytes()
            model=model_args(False,str(root/'checkpoint'))
            data=data_args(dataset_name='RSTPReid',sparse_output_dir=str(root/'sparse'),dense_output_dir=str(root/'dense'))
            training=NS(task_type='tbpr',encode_type='image',fp16=True,bf16=False,output_dir=str(root/'logs'))
            runtime=NS(encoder=NS(config=NS(model_type='qwen2_5_vl')),processor=NS())
            api=encoding.pipeline_module()
            selections=[]
            def factory(_model,_processor,_args,_data,_task,selection):
                selections.append(selection)
                return NS(last_prompts=[])
            with patch.object(qwen_vl_pipeline,'init_process',return_value=(0,1,torch.device('cpu'))), \
                    patch.object(api,'_load_retrieval_runtime',return_value=runtime), \
                    patch.object(api,'_release_retrieval_runtime'), patch.object(api,'build_dataset',return_value=NS()), \
                    patch.object(encoding,'configure_processor'), patch.object(views,'model_identity',return_value=fixtures.IDENTITY), \
                    patch.object(encoding,'source_pools',side_effect=AssertionError('explicit probabilities must not be resampled')), \
                    patch.object(qwen_vl_pipeline,'dataset_records',return_value=[]), \
                    patch.object(encoding,'AdaptiveEncoder',side_effect=factory), patch.object(encoding,'write_shard'), \
                    redirect_stdout(io.StringIO()):
                for threshold in (0,0.001):
                    options.view_probability_mass_threshold=threshold
                    encoding.run_encode(model,data,training,options)
            self.assertEqual(Path(options.view_selection_file).read_bytes(),before)
            self.assertEqual(len(selections),2)
            self.assertEqual(selections[0]['selected'],selections[1]['selected'])
            self.assertNotEqual(selections[0]['views_fingerprint'],selections[1]['views_fingerprint'])
            manifests=list((root/'sparse').rglob(views.MANIFEST))
            self.assertEqual(len(manifests),2)
            for path in manifests:
                manifest=json.loads(path.read_text(encoding='utf-8'))
                saved=Path(manifest['selection_file'])
                self.assertNotEqual(saved,Path(options.view_selection_file))
                locked=views.load_selection(saved)
                self.assertTrue(mass.is_selection(locked))
                self.assertEqual(locked['selection_split'],'train')
                key=views.selection_cache_key(fixtures.IDENTITY,'RSTPReid','train',locked['selection_modality'],
                    locked['sampling'],options.adaptive_data_root,strategy=locked[mass.FIELD])
                self.assertEqual(saved,views.selection_cache_path(data.sparse_output_dir,model.model_name_or_path,'RSTPReid',key))

    def test_thresholds_share_one_probability_computation(self):
        with temporary_directory() as directory:
            options = fixtures.options(shuffled=False)
            options.view_probability_mass = True
            options.adaptive_data_root = directory
            model, data = model_args(name=directory+'/checkpoint'), data_args(sparse_output_dir=directory+'/sparse')
            fake = NS(config=NS(model_type='qwen2_5_vl'))
            with patch.object(views,'score_group',side_effect=fixtures.synthetic_score), \
                    patch.object(views,'choose_views',wraps=views.choose_views) as choose, redirect_stdout(io.StringIO()):
                results = []
                for threshold in (0.2, 0.9, 1):
                    options.view_probability_mass_threshold = threshold
                    strategy = mass.ProbabilityMassStrategy(options, model, data)
                    results.append(strategy.choose(fake, fixtures.Processor(), fixtures.IDENTITY, 'flickr', 'test',
                        fixtures.TEXTS, fixtures.IMAGES, options, report_path=Path(directory)/f'diagnostics_{threshold}.json'))
                self.assertEqual(choose.call_count, 1)
            files = list((Path(directory)/'sparse/view_selection').rglob('view_selection.json'))
            self.assertEqual(len(files), 1)
            raw = views.load_selection(files[0], view_probability_mass=False)
            self.assertEqual(raw['version'], views.VERSION)
            self.assertEqual({r['probability_source_fingerprint'] for r in results}, {raw['fingerprint']})
            self.assertEqual(len(results[-1]['selected']), 20)

    def test_exact_index_and_retrieval_share_mode_paths_for_all_tasks(self):
        with temporary_directory() as directory:
            root = Path(directory)
            for group, dataset, kind in (('ir','flickr','image'), ('tbpr','RSTPReid','image'), ('fused','webqa','corpus'), ('fused','remuq','corpus'), ('fused','llava','corpus')):
                for pmi in (False, True):
                    env = dict(MODEL_FAMILY='qwen25', DATASET=dataset, ENCODE_TYPE=kind,
                        CHECKPOINT_ROOT=str(root/'models'), DATA_ROOT=str(root/'data'),
                        SPARSE_OUTPUT_ROOT=str(root/'sparse'), DENSE_OUTPUT_ROOT=str(root/'dense'),
                        VIEW_EXAMPLE_MODE='fixed', VIEW_SELECTION_ROUNDS='2', VIEW_TEXTS_PER_ROUND='2', VIEW_IMAGES_PER_ROUND='1',
                        USE_HIERARCHICAL_VIEWS='true', USE_VIEW_PMI=str(pmi).lower(),
                        USE_VIEW_PROBABILITY_MASS='true', VIEW_PROBABILITY_MASS_THRESHOLD='0.8', DRY_RUN='true')
                    settings = retrieval.Settings.from_environment(group, env, project=ROOT)
                    task = retrieval.job_types(settings, dataset)[0][0]
                    model, data, training = retrieval.namespaces(settings, 'qwen25', dataset, task, kind)
                    identity = fixtures.IDENTITY
                    source = fixtures.choose(settings.options, identity=identity, dataset=dataset)
                    selected = mass.derive(source, 0.8)
                    path, key, strategy = retrieval.selection_location(settings, model, data, training, identity)
                    views.atomic_json(path, selected)
                    sparse = encoding.artifact_dirs(model, data, training, selected)[1]
                    manifest = dict(encoding.metadata(model,data,task,kind,identity,selected),num_shards=1,
                        selection_file=str(path),selection_fingerprint=selected['fingerprint'])
                    views.atomic_json(sparse/views.MANIFEST,manifest)
                    (sparse/'collection').mkdir()
                    (sparse/'collection/corpus_0.jsonl').write_text('{"id":"0","vector":{"token":10}}\n',encoding='utf-8')
                    with self.subTest(group=group,dataset=dataset,pmi=pmi), redirect_stdout(io.StringIO()):
                        strategy.validate(views.validate_index_selection(sparse,manifest))
                        index_settings=indexing.Settings.from_environment(group,env,exact_encoding=True)
                        jobs, missing=indexing.discover_exact(index_settings,identity_loader=lambda *a,**k:identity)
                        self.assertEqual([job.root for job in jobs],[sparse.resolve()])
                        self.assertFalse(missing)
                    other = retrieval.Settings.from_environment(group,dict(env,VIEW_PROBABILITY_MASS_THRESHOLD='0.7'),project=ROOT)
                    other_path, _, _ = retrieval.selection_location(other,model,data,training,identity)
                    self.assertNotEqual(path,other_path)

    def test_disabled_threshold_does_not_change_original_cache_or_artifact_metadata(self):
        settings=[]
        for threshold in ('0.1','0.9'):
            settings.append(retrieval.Settings.from_environment('tbpr',dict(MODEL_FAMILY='qwen25',DATASET='RSTPReid',
                VIEW_EXAMPLE_MODE='fixed',USE_VIEW_PROBABILITY_MASS='false',VIEW_PROBABILITY_MASS_THRESHOLD=threshold),project=ROOT))
        model,data,training=retrieval.namespaces(settings[0],'qwen25','RSTPReid','tbpr','image')
        self.assertEqual(retrieval.selection_location(settings[0],model,data,training,fixtures.IDENTITY),
                         retrieval.selection_location(settings[1],model,data,training,fixtures.IDENTITY))
        source=fixtures.choose(fixtures.options(shuffled=False))
        config=encoding.encoding_configuration(model,data,'tbpr','image',fixtures.IDENTITY,source)
        self.assertNotIn(mass.FIELD,config)
        self.assertNotIn('selection_strategy',config)

    def test_pipeline_preflight_inherits_locked_count_and_rejects_threshold_conflicts(self):
        api=encoding.pipeline_module()
        with temporary_directory() as directory:
            root=Path(directory)
            source=fixtures.choose(fixtures.options(shuffled=False,hierarchical=True),dataset='RSTPReid')
            selected=mass.derive(source,0.7)
            selection_path=root/'selection.json'; views.atomic_json(selection_path,selected)
            args=api._parser().parse_args(['--dataset-name','RSTPReid','--task-type','tbpr','--query-type','text',
                '--adaptive-views','--view-selection-file',str(selection_path),
                '--view-probability-mass','true','--view-probability-mass-threshold','0.7'])
            args.retrieval_model_name_or_path=str(root/'checkpoint')
            args.output_dir=str(root/'results')
            args.stage1_sparse_index=str(root/'sparse')
            args.stage2_sparse_index=args.stage1_sparse_index
            args.stage2_dense_index=str(root/'dense')
            model,data=api._retrieval_namespaces(args,eol_type='disassembleeol_separate',calculate_type='concat',sparse_manual=False)
            manifest=dict(encoding.metadata(model,data,'tbpr','image',fixtures.IDENTITY,selected),num_shards=1,
                selection_file=str(selection_path),selection_fingerprint=selected['fingerprint'])
            for artifact in (Path(args.stage2_dense_index),Path(args.stage1_sparse_index)):
                views.atomic_json(artifact/views.MANIFEST,manifest)
            (Path(args.stage2_dense_index)/'corpus_0.pkl').touch()
            index=Path(args.stage1_sparse_index)/'collection/index'; index.mkdir(parents=True)
            (index/'segments_1').touch()
            with patch.object(encoding,'load_model_identity',return_value=fixtures.IDENTITY):
                locked=encoding.preflight(args,api)
                self.assertEqual(locked['selected'],selected['selected'])
                self.assertTrue(args.output_dir.endswith('_pm_'+selected['views_fingerprint'][:16]))
                args.view_probability_mass_threshold=0.8
                with self.assertRaisesRegex(ValueError,'threshold differs'):
                    encoding.preflight(args,api)
                args.view_probability_mass=False
                with self.assertRaisesRegex(ValueError,'mode differs'):
                    encoding.preflight(args,api)


class ModelTests(RealModelTests):
    def compare(self, model, processor, records, family, adapter='adaptive'):
        if adapter!='adaptive':
            return
        for hierarchical in (False,True):
            source=fixtures.choose(fixtures.options(shuffled=False,hierarchical=hierarchical))
            dense_reference=None
            for threshold in (0,0.7,1):
                selected=mass.derive(source,threshold)
                args=model_args(True,family)
                data=data_args()
                with self.subTest(family=family,hierarchy=hierarchical,threshold=threshold), redirect_stdout(io.StringIO()):
                    encoder=encoding.AdaptiveEncoder(model,processor,args,data,'ir',selected)
                    sparse,dense=encoder.encode(records)
                    self.assertTrue(all(len(row)==len(selected['selected']) for row in sparse))
                    self.assertTrue(all(torch.isfinite(row).all() for row in sparse))
                    self.assertEqual(dense.shape[0],len(records))
                    self.assertTrue(torch.isfinite(dense).all())
                    if dense_reference is not None:
                        self.assertTrue(torch.allclose(dense,dense_reference,atol=2e-5,rtol=2e-5))
                    dense_reference=dense


class InterfaceTests(unittest.TestCase):
    def test_python_parsers_and_conflicting_modes(self):
        from transformers import HfArgumentParser
        options=HfArgumentParser(views.AdaptiveViewArguments).parse_args_into_dataclasses([
            '--adaptive_views','true','--view_probability_mass','true','--view_probability_mass_threshold','0.7'])[0]
        self.assertTrue(mass.enabled(options))
        self.assertEqual(options.view_probability_mass_threshold,0.7)
        api=encoding.pipeline_module()
        args=api._parser().parse_args(['--dataset-name','RSTPReid','--task-type','tbpr',
            '--view-probability-mass','true'])
        self.assertTrue(mass.enabled(args))
        env=dict(USE_VIEW_PROBABILITY_MASS='true',VIEW_PROBABILITY_MASS_THRESHOLD='0.7',USE_VIEW_OVERLAP_SELECTION='true')
        with self.assertRaisesRegex(ValueError,'mutually exclusive'):
            encode_batch.encoding_controls(env)

    def test_shell_configuration_reaches_dispatchers_and_direct_python_commands(self):
        bash='D:/Git/bin/bash.exe' if os.name=='nt' else shutil.which('bash')
        if not bash or not Path(bash).is_file():
            self.skipTest('Bash unavailable')
        names=['encode_adaptive_views.sh','three_stage_retrieval_pipeline_adaptive_views.sh']
        for group in ('ir','tbpr','fused'):
            names.extend([f'encode_adaptive_views_{group}.sh',f'three_stage_retrieval_pipeline_adaptive_views_{group}.sh',
                          f'sparse_index_adaptive_{group}.sh',f'three_stage_adaptive_{group}_multi.sh'])
        # Git Bash interprets the Win32 \\?\ prefix as a UNC network path.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); scripts=root/'scripts'; scripts.mkdir()
            shutil.copyfile(ROOT/'scripts/freeret_dense_mode.sh',scripts/'freeret_dense_mode.sh')
            shutil.copyfile(ROOT / "scripts/stage2_retrieval_mode.sh", scripts / "stage2_retrieval_mode.sh")
            shutil.copyfile(ROOT / "scripts/stc_sparse_mode.sh", scripts / "stc_sparse_mode.sh")
            shutil.copyfile(ROOT / "scripts/lens_sparse_mode.sh", scripts / "lens_sparse_mode.sh")
            capture=root/'capture.sh'
            capture.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@"\nprintf \'mass=%s threshold=%s\\n\' "${USE_VIEW_PROBABILITY_MASS-unset}" "${VIEW_PROBABILITY_MASS_THRESHOLD-unset}"\n',encoding='utf-8')
            capture.chmod(0o755)
            for local in (False,True):
                for name in names:
                    with self.subTest(script=name,local=local):
                        body=(ROOT/'scripts'/name).read_text(encoding='utf-8')
                        if local:
                            body=body.replace('${USE_VIEW_PROBABILITY_MASS-false}','${USE_VIEW_PROBABILITY_MASS-true}')
                            body=re.sub(r'\$\{VIEW_PROBABILITY_MASS_THRESHOLD-[0-9.]+\}',
                                        '${VIEW_PROBABILITY_MASS_THRESHOLD-0.7}',body)
                        path=scripts/name; path.write_text(body,encoding='utf-8')
                        env={k:v for k,v in os.environ.items() if k not in {'USE_VIEW_PROBABILITY_MASS','VIEW_PROBABILITY_MASS_THRESHOLD'}}
                        env.update(PYTHON_BIN=capture.as_posix(),DEEPSPEED_BIN=capture.as_posix(),
                            MODEL_FAMILY='qwen25',DATASET='flickr',LLAVA_IMAGE_MODE='anyres',USE_FREERET_DENSE='false',
                            USE_FREERET_PROMPT_CONSTRAINTS='false',USE_FREERET_FULL_PROMPT='false',
                            VIEW_SELECTION_FILE='selection.json' if name=='three_stage_retrieval_pipeline_adaptive_views.sh' else '',
                            DENSE_CORPUS_DIR='dense',SPARSE_CORPUS_DIR='sparse',TARGET_SPARSE_DIR='',DRY_RUN='false')
                        if not local:
                            env.update(USE_VIEW_PROBABILITY_MASS='true',VIEW_PROBABILITY_MASS_THRESHOLD='0.7')
                        result=subprocess.run([bash,path.as_posix()],env=env,cwd=root,capture_output=True,text=True,timeout=30)
                        self.assertEqual(result.returncode,0,result.stderr)
                        if name in names[:2]:
                            lines=result.stdout.splitlines()
                            flag='--view_probability_mass' if name.startswith('encode') else '--view-probability-mass'
                            threshold='--view_probability_mass_threshold' if name.startswith('encode') else '--view-probability-mass-threshold'
                            self.assertEqual(lines[lines.index(flag)+1],'true')
                            self.assertEqual(lines[lines.index(threshold)+1],'0.7')
                        else:
                            self.assertIn('mass=true threshold=0.7',result.stdout)


if __name__=='__main__':
    unittest.main()
