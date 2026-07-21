import dspy

class GenerateSchemaAspects(dspy.Signature):
    """Generate 3 to 7 retrieval-useful ontology aspects from 20-30 dataset texts."""

    dataset_name: str = dspy.InputField()
    task_type: str = dspy.InputField()
    seed_texts: str = dspy.InputField()
    aspects: str = dspy.OutputField()


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

dspy.inspect_history(n=1)