"""STC is a projection of the original integer sparse vector, not a second encoder."""
from contextlib import redirect_stdout
import copy
import io
import unittest

import torch

import test_freeret_dense as fixtures
import test_stc_sparse as sparse_fixtures
import adaptive_view_encoding as encoding
import stc_sparse as stc


class ProjectionTests(unittest.TestCase):
    setUp = sparse_fixtures.STCTests.setUp
    build = sparse_fixtures.STCTests.build

    def encoder(self, *, family="qwen2_5_vl", args=None, **data):
        options = args or self.args
        options.eol_type, options.calculate_type = "disassembleeol_separate", "concat"
        return stc.SparseEncoder(self.tokenizer, fixtures.data_args(print_sparse=False, **data),
                                 options, family=family)

    def test_identity_projection_preserves_integer_weights_for_each_family_and_reducer(self):
        options = sparse_fixtures.args_at(self.root, stc_n_clusters=4)
        self.build(options)
        values = torch.tensor([[0., .171, .269, .304, .049, .126, .097, 0., 0., 0.],
                               [0., .329, .142, .296, .076, .113, .185, 0., 0., 0.]])
        for family in ("qwen2_5_vl", "llava_llama", "llava_mistral"):
            for merge in ("sum", "max", "replace"):
                for mean in (False, True):
                    for record in ({"text": "car dog"}, {"image": "fixture", "text": "car dog"}):
                        with self.subTest(family=family, merge=merge, mean=mean, image="image" in record):
                            encoder = self.encoder(family=family, args=options, sparse_length=6,
                                                   sparse_value_type=merge, sparse_value_mean=mean)
                            original = encoding.uncompressed_sparse_vectors(encoder.legacy_context, [record], [values])[0]
                            projected = encoder.vector(values, record)
                            recovered = {term: projected[f"stc_{cluster:06d}"]
                                         for term, cluster in encoder.term_to_cluster.items() if term in original}
                            self.assertEqual(recovered, original)
                            self.assertEqual(sum(projected.values()), sum(original.values()))

    def test_normalized_aliases_use_smallest_mappable_id_for_all_records(self):
        for spelling in ("CAR", "ĠCAR"):
            self.tokenizer.words = dict(type(self.tokenizer).words)
            self.tokenizer.words.pop("hound")
            self.tokenizer.words[spelling] = 4  # ID 4 belongs to another embedding cluster.
            vocabulary = self.build()
            mapping = dict(zip(vocabulary.arrays["token_ids"], vocabulary.arrays["cluster_ids"]))
            self.assertNotEqual(mapping[1], mapping[4])
            for family in ("qwen2_5_vl", "llava_llama", "llava_mistral"):
                for filtered in (False, True):
                    with self.subTest(spelling=spelling, family=family, filtered=filtered):
                        encoder = self.encoder(family=family, sparse_length=2, is_filtered=filtered)
                        self.assertEqual(encoder.term_to_cluster["car"], mapping[1])
                        values = torch.zeros(1, 10)
                        values[0, 4], values[0, 3] = .3, .25
                        original = encoding.uncompressed_sparse_vectors(encoder.legacy_context, [{}], [values],
                            allowed_token_ids=encoder.allowed_token_ids)[0]
                        self.assertEqual(encoder.vector(values), encoder.project(original))
                        self.assertEqual(sum(encoder.vector(values).values()), sum(original.values()))
                        if spelling == "CAR" or filtered:
                            self.assertEqual(encoder.vector(values)[f"stc_{mapping[1]:06d}"], 30)

    def test_excluded_alias_contribution_is_removed_before_term_merge_without_refill(self):
        self.tokenizer.words = dict(type(self.tokenizer).words)
        self.tokenizer.words.pop("hound")
        self.tokenizer.words["CAR"] = 4
        self.input[1] = 0
        vocabulary = self.build()
        cluster = int(vocabulary.arrays["cluster_ids"][list(vocabulary.arrays["token_ids"]).index(4)])
        for family in ("qwen2_5_vl", "llava_llama", "llava_mistral"):
            for merge in ("sum", "max", "replace"):
                with self.subTest(family=family, merge=merge):
                    encoder = self.encoder(family=family, sparse_length=2, sparse_value_type=merge)
                    values = torch.zeros(1, 10)
                    values[0, 1], values[0, 4], values[0, 2] = .5, .2, .19
                    self.assertEqual(encoder.vector(values), {f"stc_{cluster:06d}": 20})

    def test_case_preserving_policy_keeps_distinct_original_terms(self):
        self.tokenizer.words = dict(type(self.tokenizer).words)
        self.tokenizer.words.pop("hound")
        self.tokenizer.words["CAR"] = 4
        vocabulary = self.build()
        mapping = dict(zip(vocabulary.arrays["token_ids"], vocabulary.arrays["cluster_ids"]))
        for family in ("qwen2_5_vl", "llava_llama", "llava_mistral"):
            for filtered in (False, True):
                with self.subTest(family=family, filtered=filtered):
                    encoder = self.encoder(family=family, sparse_length=2, is_filtered=filtered,
                                           sparse_lower_or_upper="upper")
                    self.assertNotEqual(encoder.term_to_cluster["car"], encoder.term_to_cluster["CAR"])
                    values = torch.zeros(1, 10)
                    values[0, 1], values[0, 4] = .2, .3
                    self.assertEqual(encoder.vector(values), {f"stc_{mapping[1]:06d}": 20,
                                                              f"stc_{mapping[4]:06d}": 30})

    def test_unselected_members_never_increase_weight_or_refill_cluster_budget(self):
        self.build()
        encoder = self.encoder(sparse_length=2)
        encoder.term_to_cluster = dict.fromkeys(encoder.term_to_cluster, 0)
        values = torch.zeros(1, 10)
        values[0, 1:7] = torch.tensor([.9, .8, .7, .6, .5, .4])
        self.assertEqual(encoder.vector(values), {"stc_000000": 170})

    def test_non_stc_dispatch_is_unchanged_by_shared_entry(self):
        self.build()
        for family in ("qwen2_5_vl", "llava_mistral"):
            encoder = self.encoder(family=family, sparse_length=3)
            ordinary = copy.copy(encoder.legacy_context)
            ordinary.model_args = copy.copy(ordinary.model_args)
            ordinary.model_args.use_stc_sparse = False
            values = torch.tensor([[0., .101, .202, .303, 0., 0., 0., 0., 0., 0.]])
            with redirect_stdout(io.StringIO()):
                actual = encoding.sparse_vectors(ordinary, [{}], [values])
            self.assertEqual(actual, encoding.uncompressed_sparse_vectors(ordinary, [{}], [values]))


if __name__ == "__main__":
    unittest.main()
