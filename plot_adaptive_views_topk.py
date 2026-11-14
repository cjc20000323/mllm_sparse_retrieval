"""Plot Recall@1/5/10/20 against the number of adaptive perspectives.

Example (run from the project root):
    python plot_adaptive_views_topk.py

The supplied ReMuQ (llava_mistral, image + text to text) CSV files are transcribed below.
No input files are read at runtime. Each hybrid weight is plotted in a separate
figure alongside its three baselines. Dense results are retained for each weight.
Corresponding panels share the same Recall range across all five figures.
Recall and cumulative view probabilities are shown as percent;
cumulative probabilities retain two decimal places and the original scale.
For each weight, a 300-DPI PNG and a vector PDF named
adaptive_topk_recall_alpha_<weight> are saved next to this script, independently
of the working directory. Use --output-dir only to override that location.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import math
from pathlib import Path
import textwrap


METRICS = ("R@1", "R@5", "R@10", "R@20")

# Transcribed from the supplied ReMuQ (llava_mistral) view_ranking (18).csv.
# Top-K includes the first K entries; probabilities retain the original scale.
VIEW_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
VIEW_ORDER = [
    ('G', 'relations, such as spatial relations, interactions and ownership', 0.1668542032347133),
    ('F', 'actions, such as behavior, participants, tools and targets', 0.1448447228060507),
    ('A', 'scene, such as location, indoor or outdoor setting and terrain', 0.13543174666668262),
    ('H', 'visual attributes, such as color, shape, size and material', 0.13400293907445204),
    ('I', 'spatial layout, such as foreground and background, distance and direction', 0.12381357491233896),
    ('B', 'natural environment, such as weather, lighting, season and time', 0.09862656544624132),
    ('D', 'clothing, such as category, color, material and style', 0.07855963142181208),
    ('C', 'people, such as number of people, appearance and pose', 0.06904182820394125),
    ('J', 'high-level semantics, such as events, activities and functions', 0.04489433273390483),
    ('E', 'objects, such as category, number, appearance and state', 0.0039304554998628935),
]

# Recall fractions from the supplied ReMuQ (llava_mistral) topk_summary (18).csv (200 rows).
# Sparse baselines are shared; Dense and Hybrid retain all five alpha values.
RECALL_SERIES = {
    ('stage1', 'sparse_concat', None): {
        'R@1': [0.18703241895261846, 0.29925187032418954, 0.3577168190634525, 0.3458021612635079,
                0.3230811859240787, 0.3117206982543641, 0.3089498476032142, 0.29842061512884455,
                0.29260182876142976, 0.29010806317539484],
        'R@5': [0.3325020781379884, 0.48212801330008315, 0.5497367691881407, 0.540592962039346,
                0.5217511776115268, 0.5081740094208922, 0.4979218620116376, 0.4915489055139928,
                0.4882238847326129, 0.4829592684954281],
        'R@10': [0.4031587697423109, 0.5550013854253256, 0.6350789692435578, 0.6145746744250485,
                 0.5890828484344693, 0.5876974231088944, 0.5685785536159601, 0.569963978941535,
                 0.5594347464671654, 0.5605430867276254],
        'R@20': [0.4704904405652535, 0.6320310335272928, 0.7071210861734553, 0.6896647270712108,
                 0.6630645608201718, 0.6636187309504018, 0.6422831809365476, 0.6422831809365476,
                 0.6320310335272928, 0.6323081185924079],
    },
    ('stage2', 'dense', 0.5): {
        'R@1': [0.21557218065946246, 0.25353283458021614, 0.28983097811027986, 0.3003602105846495,
                0.30479357162648935, 0.30673316708229426, 0.32363535605430865, 0.313937378775284,
                0.33194790800775836, 0.3333333333333333],
        'R@5': [0.4596841230257689, 0.514824050983652, 0.5641451925741202, 0.5680243834857301,
                0.5755056802438349, 0.5755056802438349, 0.5843724023275145, 0.5816015516763646,
                0.5885286783042394, 0.5957328899972292],
        'R@10': [0.5455804932114159, 0.6178996952064284, 0.6683291770573566, 0.6741479634247715,
                 0.6708229426433915, 0.6716541978387365, 0.6810750900526462, 0.683291770573566,
                 0.6844001108340261, 0.6918814075921308],
        'R@20': [0.6251039068994181, 0.7115544472152951, 0.7608755888057633, 0.7619839290662234,
                 0.7542255472430036, 0.7550568024383486, 0.7586589082848434, 0.7572734829592684,
                 0.7583818232197285, 0.7653089498476032],
    },
    ('stage2', 'dense', 0.6): {
        'R@1': [0.21557218065946246, 0.25353283458021614, 0.28983097811027986, 0.3003602105846495,
                0.30479357162648935, 0.30673316708229426, 0.32363535605430865, 0.313937378775284,
                0.33194790800775836, 0.3333333333333333],
        'R@5': [0.4596841230257689, 0.514824050983652, 0.5641451925741202, 0.5680243834857301,
                0.5755056802438349, 0.5755056802438349, 0.5843724023275145, 0.5816015516763646,
                0.5885286783042394, 0.5957328899972292],
        'R@10': [0.5455804932114159, 0.6178996952064284, 0.6683291770573566, 0.6741479634247715,
                 0.6708229426433915, 0.6716541978387365, 0.6810750900526462, 0.683291770573566,
                 0.6844001108340261, 0.6918814075921308],
        'R@20': [0.6251039068994181, 0.7115544472152951, 0.7608755888057633, 0.7619839290662234,
                 0.7542255472430036, 0.7550568024383486, 0.7586589082848434, 0.7572734829592684,
                 0.7583818232197285, 0.7653089498476032],
    },
    ('stage2', 'dense', 0.7): {
        'R@1': [0.21557218065946246, 0.25353283458021614, 0.28983097811027986, 0.3003602105846495,
                0.30479357162648935, 0.30673316708229426, 0.32363535605430865, 0.313937378775284,
                0.33194790800775836, 0.3333333333333333],
        'R@5': [0.4596841230257689, 0.514824050983652, 0.5641451925741202, 0.5680243834857301,
                0.5755056802438349, 0.5755056802438349, 0.5843724023275145, 0.5816015516763646,
                0.5885286783042394, 0.5957328899972292],
        'R@10': [0.5455804932114159, 0.6178996952064284, 0.6683291770573566, 0.6741479634247715,
                 0.6708229426433915, 0.6716541978387365, 0.6810750900526462, 0.683291770573566,
                 0.6844001108340261, 0.6918814075921308],
        'R@20': [0.6251039068994181, 0.7115544472152951, 0.7608755888057633, 0.7619839290662234,
                 0.7542255472430036, 0.7550568024383486, 0.7586589082848434, 0.7572734829592684,
                 0.7583818232197285, 0.7653089498476032],
    },
    ('stage2', 'dense', 0.8): {
        'R@1': [0.21557218065946246, 0.25353283458021614, 0.28983097811027986, 0.3003602105846495,
                0.30479357162648935, 0.30673316708229426, 0.32363535605430865, 0.313937378775284,
                0.33194790800775836, 0.3333333333333333],
        'R@5': [0.4596841230257689, 0.514824050983652, 0.5641451925741202, 0.5680243834857301,
                0.5755056802438349, 0.5755056802438349, 0.5843724023275145, 0.5816015516763646,
                0.5885286783042394, 0.5957328899972292],
        'R@10': [0.5455804932114159, 0.6178996952064284, 0.6683291770573566, 0.6741479634247715,
                 0.6708229426433915, 0.6716541978387365, 0.6810750900526462, 0.683291770573566,
                 0.6844001108340261, 0.6918814075921308],
        'R@20': [0.6251039068994181, 0.7115544472152951, 0.7608755888057633, 0.7619839290662234,
                 0.7542255472430036, 0.7550568024383486, 0.7586589082848434, 0.7572734829592684,
                 0.7583818232197285, 0.7653089498476032],
    },
    ('stage2', 'dense', 0.9): {
        'R@1': [0.21557218065946246, 0.25353283458021614, 0.28983097811027986, 0.3003602105846495,
                0.30479357162648935, 0.30673316708229426, 0.32363535605430865, 0.313937378775284,
                0.33194790800775836, 0.3333333333333333],
        'R@5': [0.4596841230257689, 0.514824050983652, 0.5641451925741202, 0.5680243834857301,
                0.5755056802438349, 0.5755056802438349, 0.5843724023275145, 0.5816015516763646,
                0.5885286783042394, 0.5957328899972292],
        'R@10': [0.5455804932114159, 0.6178996952064284, 0.6683291770573566, 0.6741479634247715,
                 0.6708229426433915, 0.6716541978387365, 0.6810750900526462, 0.683291770573566,
                 0.6844001108340261, 0.6918814075921308],
        'R@20': [0.6251039068994181, 0.7115544472152951, 0.7608755888057633, 0.7619839290662234,
                 0.7542255472430036, 0.7550568024383486, 0.7586589082848434, 0.7572734829592684,
                 0.7583818232197285, 0.7653089498476032],
    },
    ('stage2', 'sparse', None): {
        'R@1': [0.18703241895261846, 0.29925187032418954, 0.3577168190634525, 0.3458021612635079,
                0.3230811859240787, 0.3117206982543641, 0.3089498476032142, 0.29842061512884455,
                0.29260182876142976, 0.29010806317539484],
        'R@5': [0.3325020781379884, 0.48212801330008315, 0.5497367691881407, 0.540592962039346,
                0.5217511776115268, 0.5081740094208922, 0.4979218620116376, 0.4915489055139928,
                0.4882238847326129, 0.4829592684954281],
        'R@10': [0.4031587697423109, 0.5550013854253256, 0.6350789692435578, 0.6145746744250485,
                 0.5890828484344693, 0.5876974231088944, 0.5685785536159601, 0.569963978941535,
                 0.5594347464671654, 0.5605430867276254],
        'R@20': [0.4704904405652535, 0.6320310335272928, 0.7071210861734553, 0.6896647270712108,
                 0.6630645608201718, 0.6636187309504018, 0.6422831809365476, 0.6422831809365476,
                 0.6320310335272928, 0.6323081185924079],
    },
    ('stage2', 'hybrid', 0.5): {
        'R@1': [0.25353283458021614, 0.37378775284012195, 0.456359102244389, 0.4549736769188141,
                0.44693821003047934, 0.4314214463840399, 0.42975893599334997, 0.4192297035189803,
                0.43197561651426986, 0.4316985314491549],
        'R@5': [0.42615683014685507, 0.5857578276530895, 0.6672208367968967, 0.6658354114713217,
                0.6489332224993073, 0.6386810750900527, 0.6367414796342478, 0.6384039900249376,
                0.6359102244389028, 0.6406206705458576],
        'R@10': [0.5042948185092824, 0.6683291770573566, 0.7406483790523691, 0.7364921030756443,
                 0.7193128290385148, 0.7115544472152951, 0.7118315322804101, 0.7126627874757551,
                 0.7121086173455251, 0.7096148517594901],
        'R@20': [0.5796619562205597, 0.7387087835965641, 0.8024383485730119, 0.7952341368800222,
                 0.7863674147963424, 0.7802715433638127, 0.771958991410363, 0.7758381823219729,
                 0.7694652258243281, 0.7725131615405929],
    },
    ('stage2', 'hybrid', 0.6): {
        'R@1': [0.27957883070102524, 0.39623164311443615, 0.4752008866722084, 0.48517594901634803,
                0.47492380160709335, 0.4632862288722638, 0.46633416458852867, 0.456913272374619,
                0.47187586589082847, 0.4790800775838182],
        'R@5': [0.45857578276530897, 0.6198392906622333, 0.6927126627874758, 0.6932668329177057,
                0.6805209199224161, 0.6744250484898864, 0.6722083679689664, 0.6711000277085065,
                0.6738708783596564, 0.6807980049875312],
        'R@10': [0.5333887503463564, 0.6902188972014408, 0.7633693543917983, 0.7575505680243835,
                 0.742865059573289, 0.7395400387919091, 0.7384316985314492, 0.742865059573289,
                 0.7403712939872541, 0.7403712939872541],
        'R@20': [0.6082017179274037, 0.7522859517871987, 0.8187863674147964, 0.8135217511776115,
                 0.8032696037683569, 0.7977279024660571, 0.7896924355777224, 0.7924632862288723,
                 0.7844278193405375, 0.7930174563591023],
    },
    ('stage2', 'hybrid', 0.7): {
        'R@1': [0.28706012745912995, 0.39927957883070103, 0.47270712108617347, 0.48129675810473815,
                0.4801884178442782, 0.47575505680243835, 0.4826821834303131, 0.4696591853699086,
                0.48600720421169297, 0.4882238847326129],
        'R@5': [0.48794679966749793, 0.6275976724854531, 0.7068440011083402, 0.7029648101967304,
                0.6974231088944306, 0.6902188972014408, 0.6946522582432807, 0.6940980881130507,
                0.6982543640897756, 0.7004710446106955],
        'R@10': [0.5616514269880853, 0.7013022998060404, 0.773067331670823, 0.7702964810196731,
                 0.7600443336104183, 0.7545026323081185, 0.7536713771127737, 0.7578276530894985,
                 0.7536713771127737, 0.7569963978941535],
        'R@20': [0.628428927680798, 0.7608755888057633, 0.8251593239124411, 0.8212801330008312,
                 0.8085342200055416, 0.8068717096148518, 0.799667497921862, 0.8016070933776669,
                 0.7938487115544473, 0.800498753117207],
    },
    ('stage2', 'hybrid', 0.8): {
        'R@1': [0.2804100858963702, 0.3660293710169022, 0.4322527015793849, 0.4513715710723192,
                0.44749238016070936, 0.4427819340537545, 0.45580493211415907, 0.44111942366306456,
                0.46162371848157385, 0.4688279301745636],
        'R@5': [0.5018010529232474, 0.6178996952064284, 0.6924355777223608, 0.6902188972014408,
                0.6877251316154059, 0.6807980049875312, 0.6838459407037961, 0.6827376004433361,
                0.686616791354946, 0.6921584926572458],
        'R@10': [0.5768911055694098, 0.7010252147409255, 0.7678027154336381, 0.769188140759213,
                 0.7605985037406484, 0.7547797173732336, 0.7556109725685786, 0.7561651426988085,
                 0.7503463563313937, 0.7592130784150735],
        'R@20': [0.6361873095040177, 0.7611526738708784, 0.8212801330008312, 0.8151842615683015,
                 0.8052091992241618, 0.8052091992241618, 0.800498753117207, 0.8038237738985868,
                 0.7952341368800222, 0.8032696037683569],
    },
    ('stage2', 'hybrid', 0.9): {
        'R@1': [0.257134940426711, 0.31338320864505403, 0.36436686062621226, 0.3773898586866168,
                0.38126904959822666, 0.37849819894707676, 0.3967858132446661, 0.3798836242726517,
                0.4034358548074259, 0.4062067054585758],
        'R@5': [0.49570518149071763, 0.5863119977833194, 0.6492103075644223, 0.6489332224993073,
                0.6505957328899973, 0.6458852867830424, 0.6517040731504572, 0.6519811582155722,
                0.6544749238016071, 0.6616791354945969],
        'R@10': [0.5766140205042948, 0.6724854530340815, 0.7359379329454142, 0.7364921030756443,
                 0.7339983374896093, 0.7276253809919645, 0.7323358270989194, 0.7320587420338044,
                 0.7301191465779995, 0.7387087835965641],
        'R@20': [0.6356331393737877, 0.743419229703519, 0.8002216680520919, 0.799667497921862,
                 0.7924632862288723, 0.7924632862288723, 0.7891382654474923, 0.7905236907730673,
                 0.7866444998614575, 0.7916320310335273],
    },
}
EXPERIMENT = {
    'model_family': 'llava_mistral',
    'model': '/root/gpufree-share/checkpoints/llava-hf-llava-v1.6-mistral-7b-hf',
    'dataset': 'remuq',
    'split': 'test',
    'task': 'it2t',
    'query_type': 'image',
    'execution_mode': 'two_stage',
    'category': 'all',
    'query_count': '3609',
    'hybrid_alphas': '0.5, 0.6, 0.7, 0.8, 0.9',
}


SERIES_STYLES = {
    ("stage1", "sparse_concat", None): ("Stage 1 - Sparse", "#0072B2", "-", "o", 8, "white", 2.6),
    ("stage2", "dense", None): ("Stage 2 - Dense", "#009E73", "-", "s", 5, "#009E73", 1.8),
    ("stage2", "sparse", None): ("Stage 2 - Sparse", "#E69F00", "--", "x", 5, "#E69F00", 1.8),
    ("stage2", "hybrid", 0.5): ("Hybrid (alpha = 0.5)", "#D55E00", "-", "D", 5, "#D55E00", 1.7),
    ("stage2", "hybrid", 0.6): ("Hybrid (alpha = 0.6)", "#CC79A7", "-", "^", 5, "#CC79A7", 1.7),
    ("stage2", "hybrid", 0.7): ("Hybrid (alpha = 0.7)", "#56B4E9", "-", "v", 5, "#56B4E9", 1.7),
    ("stage2", "hybrid", 0.8): ("Hybrid (alpha = 0.8)", "#6A3D9A", "-", "P", 5, "#6A3D9A", 1.7),
    ("stage2", "hybrid", 0.9): ("Hybrid (alpha = 0.9)", "#333333", "-", "*", 7, "#333333", 1.7),
}


@dataclass(frozen=True)
class Perspective:
    rank: int
    label: str
    aspect: str
    probability: float

    @property
    def name(self):
        return self.aspect.split(", such as", 1)[0]


@dataclass
class PlotData:
    perspectives: tuple[Perspective, ...]
    metadata: dict[str, str]
    # A None alpha marks a shared series; a numeric alpha belongs to one figure.
    curves: dict[tuple[str, str, float | None], dict[int, tuple[float, ...]]]
    max_views: int

    @property
    def hybrid_alphas(self):
        return tuple(sorted(key[2] for key in self.curves
                            if key[:2] == ("stage2", "hybrid") and key[2] is not None))

    @property
    def cumulative_probabilities(self):
        return tuple(math.fsum(item.probability for item in self.perspectives[:count])
                     for count in range(1, len(self.perspectives) + 1))


def build_plot_data():
    """Validate and arrange the in-code lists; no external data is loaded."""
    if not VIEW_COUNTS or VIEW_COUNTS != list(range(1, len(VIEW_COUNTS) + 1)):
        raise ValueError("VIEW_COUNTS must be consecutive integers starting at 1")
    if len(VIEW_ORDER) < len(VIEW_COUNTS):
        raise ValueError("VIEW_ORDER must cover every VIEW_COUNTS entry")
    perspectives = tuple(Perspective(rank, label, aspect, probability)
                         for rank, (label, aspect, probability) in enumerate(VIEW_ORDER, 1))
    labels = [item.label for item in perspectives]
    if len(set(labels)) != len(labels) or any(not item.label or not item.aspect for item in perspectives):
        raise ValueError("VIEW_ORDER labels must be unique; labels and aspects must not be empty")
    probabilities = [item.probability for item in perspectives]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("Perspective probabilities must be finite fractions in [0, 1]")
    if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("The complete perspective ranking must sum to probability 1")
    if any(first < second for first, second in zip(probabilities, probabilities[1:])):
        raise ValueError("Perspective probabilities must follow decreasing rank order")
    if not RECALL_SERIES:
        raise ValueError("RECALL_SERIES must contain at least one series")
    curves = {}
    for key, metrics in RECALL_SERIES.items():
        for metric in METRICS:
            values = metrics.get(metric)
            if values is None or len(values) != len(VIEW_COUNTS):
                raise ValueError(f"{key}, {metric}: expected {len(VIEW_COUNTS)} Recall values")
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                raise ValueError(f"{key}, {metric}: Recall values must be fractions in [0, 1]")
        curves[key] = {count: tuple(metrics[metric][index] for metric in METRICS)
                       for index, count in enumerate(VIEW_COUNTS)}
    order = {key: index for index, key in enumerate(SERIES_STYLES)}
    curves = dict(sorted(curves.items(), key=lambda pair: (
        order.get(pair[0], order.get((*pair[0][:2], None), len(order))),
        pair[0][:2], -1 if pair[0][2] is None else pair[0][2])))
    return PlotData(perspectives, dict(EXPERIMENT), curves, max(VIEW_COUNTS))


def series_style(key, index=0):
    alpha_label = f" (alpha = {key[2]:.1f})" if key[2] is not None else ""
    fallback = (f"{key[0]} - {key[1]}{alpha_label}", f"C{index % 10}", ":", "v", 5, "white", 1.8)
    return SERIES_STYLES.get(key, SERIES_STYLES.get((*key[:2], None), fallback))


def overlap_notes(data):
    groups = []
    for key, points in data.curves.items():
        for group in groups:
            other = data.curves[group[0]]
            if points.keys() == other.keys() and all(
                    math.isclose(first, second, rel_tol=0, abs_tol=1e-12)
                    for count in points for first, second in zip(points[count], other[count])):
                group.append(key)
                break
        else:
            groups.append([key])
    return ["Overlapping curves: " + " = ".join(series_style(key)[0] for key in group)
            for group in groups if len(group) > 1]


def plot_notes(data):
    notes = overlap_notes(data)
    if data.max_views < len(data.perspectives):
        notes.append(f"Results available for K=1..{data.max_views}; higher-K results were not supplied.")
    return notes


def curves_for_alpha(data, alpha):
    """Select this weight's recorded curves and the shared baselines."""
    hybrid_key = ("stage2", "hybrid", alpha)
    if hybrid_key not in data.curves:
        raise ValueError(f"No Hybrid results available for alpha = {alpha}")
    return {key: points for key, points in data.curves.items()
            if key[2] is None or key[2] == alpha}


def build_figure(data, alpha):
    """Plot one weight and its baselines, with ranges from all weights."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    curves = curves_for_alpha(data, alpha)
    counts = list(range(1, data.max_views + 1))
    perspectives = data.perspectives
    cumulative = data.cumulative_probabilities
    footer_rows = math.ceil(len(perspectives) / 2)
    notes = plot_notes(replace(data, curves=curves))
    if any(set(points) != set(counts) for points in curves.values()):
        notes.append("Missing observations are shown as gaps; no interpolation is applied.")
    height = 10.15 + 0.25 * footer_rows + 0.18 * max(0, len(notes) - 1)
    bottom = (1.63 + 0.25 * footer_rows) / height
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.titlesize": 13, "axes.labelsize": 11}):
        fig, axes = plt.subplots(2, 2, figsize=(14.8, height), sharex=True)
        fig.subplots_adjust(left=0.07, right=0.98, top=0.835 - 0.017 * max(0, len(notes) - 1),
                            bottom=bottom, wspace=0.19, hspace=0.55)
        meta = data.metadata
        directions = {"tbpr": "Text to image", "t2it": "Text to image + text", "it2t": "Image + text to text"}
        direction = directions.get(meta["task"], "Image to text" if meta["query_type"] == "image" else "Text to image")
        query_details = f"{int(meta['query_count']):,} queries"
        dataset_label = {"webqa": "WebQA", "llava": "LLaVA", "flickr": "Flickr", "coco": "COCO"}.get(
            meta["dataset"], meta["dataset"])
        fig.suptitle(f"{dataset_label} | {meta['model_family']} | {direction}", y=0.985, fontsize=16, weight="semibold")
        fig.text(0.5, 0.947, f"{meta['split']} / {meta['category']} | {meta['execution_mode'].replace('_', ' ')}"
                 f" | hybrid alpha = {alpha:.1f} (dense weight) | {query_details}",
                 ha="center", fontsize=10, color="#555555")
        for metric_index, (metric, ax) in enumerate(zip(METRICS, axes.flat)):
            # Use all weights to keep this metric's limits identical across figures.
            available = [points[count][metric_index] * 100
                         for points in data.curves.values() for count in counts
                         if count in points and math.isfinite(points[count][metric_index])]
            for index, (key, points) in enumerate(curves.items()):
                label, color, linestyle, marker, size, face, width = series_style(key, index)
                y = [points[count][metric_index] * 100 if count in points else math.nan for count in counts]
                ax.plot(counts, y, label=label, color=color, linestyle=linestyle, marker=marker,
                        markersize=size, markerfacecolor=face, markeredgewidth=1.3, linewidth=width)
            padding = max(0.5, (max(available) - min(available)) * 0.12)
            ax.set_ylim(max(0, min(available) - padding), min(100, max(available) + padding))
            ax.set_xlim(0.7, data.max_views + 0.3)
            ax.set_xticks(counts, [f"{count}\n+{perspectives[count - 1].label}\n{cumulative[count - 1]:.2%}" for count in counts])
            ax.tick_params(axis="x", labelbottom=True, labelsize=9, pad=6)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.set(title=metric, xlabel="Number of perspectives (K)", ylabel="Recall (%)")
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, alpha=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels,
                   loc="upper center", bbox_to_anchor=(0.5, 0.925),
                   ncol=4, frameon=False, fontsize=10, columnspacing=2.1, labelspacing=0.7)
        for index, note in enumerate(notes):
            fig.text(0.5, 0.88 - index * 0.017, note, ha="center", fontsize=9, color="#555555")
        heading = bottom - 0.10
        fig.text(0.07, heading, "Top-K uses the first K views; +letter = newly added view; % = cumulative probability.",
                 fontsize=10, weight="semibold")
        for index, perspective in enumerate(perspectives):
            column, row = divmod(index, footer_rows)
            unreported = perspective.rank > data.max_views
            suffix = f" (no Top-{perspective.rank} result)" if unreported else ""
            fig.text(0.07 + column * 0.47, heading - 0.035 - row * (0.25 / height),
                     textwrap.fill(f"{perspective.rank:>2}. {perspective.label}  {perspective.name}{suffix}", width=65),
                     fontsize=9.5, va="top", linespacing=1.2, color="#777777" if unreported else "black")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Output directory (default: the directory containing this script)")
    args = parser.parse_args(argv)
    try:
        data = build_plot_data()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        alphas = data.hybrid_alphas
        if not alphas:
            raise ValueError("No Hybrid weights available to plot")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for alpha in alphas:
            fig = build_figure(data, alpha)
            try:
                for extension in ("png", "pdf"):
                    path = args.output_dir / f"adaptive_topk_recall_alpha_{alpha:.1f}.{extension}"
                    fig.savefig(path, dpi=300, facecolor="white")
                    paths.append(path)
            finally:
                plt.close(fig)
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    print(f"Plotted {len(alphas)} separate alpha figures, each with the baselines and one Hybrid curve.")
    print(f"Hybrid weights: {', '.join(f'{alpha:.1f}' for alpha in alphas)}; "
          f"K=1..{data.max_views}; Recall is shown in percent with shared limits per metric.")
    print("Perspective order: " + " -> ".join(item.label for item in data.perspectives))
    print("Cumulative probabilities: " + ", ".join(
        f"K={count}: {data.cumulative_probabilities[count - 1]:.2%}" for count in range(1, data.max_views + 1)))
    notes = dict.fromkeys(
        note for alpha in alphas
        for note in plot_notes(replace(data, curves=curves_for_alpha(data, alpha))))
    for note in notes:
        print(note)
    for path in paths:
        print(f"Saved: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
