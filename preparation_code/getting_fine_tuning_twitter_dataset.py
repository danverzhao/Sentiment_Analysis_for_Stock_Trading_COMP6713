from datasets import load_dataset

dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")


for split, data in dataset.items():
    data.to_csv(f"my-dataset-{split}.csv", index = None)
