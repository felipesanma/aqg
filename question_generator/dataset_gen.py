from datasets import load_dataset
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.utils import shuffle


train_dataset = load_dataset("squad", split="train")
valid_dataset = load_dataset("squad", split="validation")

df_train = pd.DataFrame(columns=["context", "answer", "question"])
df_validation = pd.DataFrame(columns=["context", "answer", "question"])


count_long = 0
count_short = 0


for index, val in enumerate(tqdm(train_dataset)):
    print(index)
    passage = val["context"]
    question = val["question"]
    answer = val["answers"]["text"][0]
    no_of_words = len(answer.split())
    if no_of_words >= 7:
        count_long = count_long + 1
        continue
    else:
        df_train.loc[count_short] = [passage] + [answer] + [question]
        count_short = count_short + 1

print("count_long train dataset: ", count_long)
print("count_short train dataset: ", count_short)


count_long = 0
count_short = 0


for index, val in enumerate(tqdm(valid_dataset)):
    print(index)
    passage = val["context"]
    question = val["question"]
    answer = val["answers"]["text"][0]
    no_of_words = len(answer.split())
    if no_of_words >= 7:
        count_long = count_long + 1
        continue
    else:
        df_validation.loc[count_short] = [passage] + [answer] + [question]
        count_short = count_short + 1

print("count_long validation dataset: ", count_long)
print("count_short validation dataset: ", count_short)

df_train = shuffle(df_train)
df_validation = shuffle(df_validation)

train_save_path = "squad_t5_train.csv"
validation_save_path = "squad_t5_validaton.csv"
df_train.to_csv(train_save_path, index=False)
df_validation.to_csv(validation_save_path, index=False)
