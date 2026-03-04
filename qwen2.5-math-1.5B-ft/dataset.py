from datasets import load_dataset

def process_data(train_path, test_path):
    # 加载数据集
    dataset = load_dataset('json', data_files={
        'train': train_path,
        'test': test_path
    })

    # 定义模板格式
    def format_example(example):
        return {
            "text": f"Instruction: {example['input']}\nAnswer: {example['output']}<|endoftext|>"
        }

    return dataset.map(format_example)