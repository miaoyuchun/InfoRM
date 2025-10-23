from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_data(data, input_template=None, input_key="input", label_key=None, apply_chat_template=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)

    # for Reinforced Fine-tuning
    label = "" if label_key is None else data[label_key]
    return prompt, label


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
        prompt_max_len=1024,
        num_processors=8,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.label_key = getattr(self.strategy.args, "label_key", None)
        self.class_key = getattr(self.strategy.args, "class_key", None)

        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template

        self.prompt_max_len = prompt_max_len

        # self.prompts = []
        # for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
        #     prompt = preprocess_data(data, self.input_template, self.input_key, self.apply_chat_template)
        #     self.prompts.append(prompt)

        processed_dataset = dataset.map(
            self.process_data, 
            remove_columns=dataset.column_names,
            num_proc=num_processors,
        )

        original = len(processed_dataset)
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)
        self.strategy.print("debug info: {} samples are filtered due to prompt length".format(len(processed_dataset) - original))
        
        self.prompts = processed_dataset["prompt"]
        self.labels = processed_dataset["label"]
        self.class_ = processed_dataset["class"]
    
    def process_data(self, data):
        prompt, label = preprocess_data(data, self.input_template, self.input_key, self.label_key, self.apply_chat_template)
        prompt_token = self.tokenizer(prompt, max_length=self.prompt_max_len, padding=False, truncation=True, return_tensors="pt", add_special_tokens=False)
        prompt_id_length = prompt_token["attention_mask"].int().sum().item()
        if prompt_id_length >= self.prompt_max_len:
            prompt = None
        if self.class_key:
            class_ = data[self.class_key]
        else:
            class_ = ""
        return {'prompt': prompt, 'label': label, 'class': class_}
        

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx], self.labels[idx], self.class_[idx]
