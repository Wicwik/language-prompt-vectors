import numpy as np

from transformers import pipeline

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from metrics.utils import binary_reverse


def predict(test_dataset, model, tokenizer, labels_list):
    y_pred = []
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=16,
        do_sample=False,
        top_p=None,
        temperature=None,
        device="cuda",
    )

    for x_test in tqdm(test_dataset["text"]):

        result = pipe(x_test)
        answer = (
            result[0]["generated_text"]
            .split("label:<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[-1]
            .strip()
        )

        for label in labels_list:
            if label.lower() == answer.lower():
                y_pred.append(label)
                break
        else:
            y_pred.append("none")
            # print(answer)

    return y_pred


def evaluate(y_pred, y_true, mapping, prefix="eval"):
    def map_func(x):
        return mapping.get(x, -1)

    print(y_pred)
    y_pred_mapped = np.vectorize(map_func)(y_pred)
    y_true_mapped = np.vectorize(map_func)(y_true)

    unique_labels = list(set(y_true_mapped))

    accuracy = accuracy_score(y_pred=y_pred_mapped, y_true=y_true_mapped)

    if len(unique_labels) > 2:
        f1 = f1_score(
            y_pred=y_pred_mapped,
            y_true=y_true_mapped,
            labels=unique_labels,
            average="macro",
        )
    else:
        invalid_idx_mask = y_pred_mapped == -1
        y_pred_mapped[invalid_idx_mask] = binary_reverse(
            y_true_mapped[invalid_idx_mask], unique_labels
        )

        f1 = f1_score(
            y_pred=y_pred_mapped,
            y_true=y_true_mapped,
            labels=unique_labels,
            pos_label=unique_labels[1],
        )

    return {f"{prefix}/accuracy": accuracy, f"{prefix}/f1": f1}
