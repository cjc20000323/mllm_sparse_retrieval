import dspy
from torch._C import dtype
import random


class GenerateSchemaAspects(dspy.Signature):
    """Generate 3 to 7 retrieval-useful ontology aspects from 20-30 dataset texts."""

    dataset_name: str = dspy.InputField()
    task_type: str = dspy.InputField()
    seed_texts: str = dspy.InputField()
    aspects: list[str] = dspy.OutputField(
        desc="3 to 7 lowercase aspect names, e.g. ['color', 'object category', 'scene']",
        min_length=3,
        max_length=7,
    )


class SchemaAspectProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateSchemaAspects)

    def forward(self, dataset_name, task_type, seed_texts):
        return self.generate(
            dataset_name=dataset_name,
            task_type=task_type,
            seed_texts=seed_texts,
        )


lm = dspy.LM(
    "openai/Meta-Llama-3.1-8B-Instruct",
    api_base="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
    temperature=0.0,
    max_tokens=256,
)

dspy.configure(lm=lm)

program = SchemaAspectProgram()

result = program(
    dataset_name="flickr",
    task_type="itr",
    seed_texts="sentence 1\nsentence 2\nsentence 3",
)


def dspy_metric(example, pred, trace=None):
    aspects = pred.aspects
    print(aspects)
    return random.random()


optimizer = dspy.MIPROv2(
    metric=dspy_metric,
    auto=None,
    num_candidates=8,
    max_bootstrapped_demos=0,
    max_labeled_demos=0,    # 如果只想优化 instruction，不加 few-shot
    verbose=True,
    track_stats=True,
    num_threads=1,
    seed=42,
)

seed_text = (
        'You are an experienced knowledge engineer and you are modeling schemas for knowledge graph construction. '
        'Given a set of sentences, you need to give several proper words or phrases for the abstract schemas of entities, relations and events in these sentences.'
        'You must return your answer in the following format: 1. phrases1\n2.phrases2\n3.phrases3\n...'
        'You can\'t return anything other than answers.'
        'These abstract intention words should fulfill the following requirements.'
        '1. The abstract schemas phrases can well represent the entities, relations and events, and it could be the type of the entities, relations and events or the related concepts of the entities, relations and events.'
        '2. Strictly follow the provided format, do not add extra characters or words.'
        '3. Write 3 to 7 word or phrase items at the highest possible abstract level if possible.'
        '4. Do not repeat the same word and the input in the answer.'
        '5. Stop immediately if you can\'t think of any more phrases, and no explanation is needed.'
        '6. Strictly limit the sum of answers between 3 and 7 items.'
        '\n'
        '\n')

trainset = [
    dspy.Example(
        dataset_name="flickr",
        task_type="image-text retrieval",
        seed_texts=seed_text,
        eval_split="dev_small",
    ).with_inputs("dataset_name", "task_type", "seed_texts"),
]

compiled = optimizer.compile(
    program,
    trainset=trainset,
    valset=trainset,
    num_trials=20,  # 主要迭代轮次：评估 12 组候选 prompt 参数
    minibatch=False,  # 每轮用完整 valset；你的 Recall@K 场景建议这样
    seed=42,
)

prediction = compiled(
    dataset_name="flickr",
    task_type="itr",
    seed_texts=seed_text,
)

print(prediction)

seed_text = """
        You are an experienced knowledge engineer and you are modeling schemas for knowledge graph construction.
        Given a set of sentences, you need to give several proper words or phrases for the abstract schemas of entities, relations and events in these sentences.
        You must return your answer in the following format: 1. phrases1\n2.phrases2\n3.phrases3\n...
        You can\'t return anything other than answers.
        These abstract intention words should fulfill the following requirements.
        1. The abstract schemas phrases can well represent the entities, relations and events, and it could be the type of the entities, relations and events or the related concepts of the entities, relations and events.
        2. Strictly follow the provided format, do not add extra characters or words.
        3. Write 3 to 7 word or phrase items at the highest possible abstract level if possible.
        4. Do not repeat the same word and the input in the answer.
        5. Stop immediately if you can\'t think of any more phrases, and no explanation is needed.
        6. Strictly limit the sum of answers between 3 and 7 items.
        \n
        \n
        Input sentences:\n<sent>\n
        """

print(seed_text)
