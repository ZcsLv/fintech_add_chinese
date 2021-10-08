# coding=utf-8
# Copyright 2020 George Mihaila.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Deal with model inference.推断 推理"""
"""https://huggingface.co/distilbert-base-uncased
    https://huggingface.co/bert-base-uncased
"""
"""
对这两个bert模型的操作
distilbert-base-uncased.pickle
distilroberta-base.pickle
"""
import pickle
import torch
#gc模块提供可选的垃圾回收器的接口
import gc
import sys
import numpy as np


def softmax(vector):
    r"""
    calculate the softmax of a vector

    Used fom: https://machinelearningmastery.com/softmax-activation-function-with-python/
    '"""

    e = np.exp(vector)

    return e / e.sum()


def inference_transformer(model_pickle_path, text_input, ids_labels):
    r"""
    Load model and tokenizer form .pickle and perform prediction using text input.
    """
    #得到tokenizer类似于分词器，model模型这俩个类
    with open(model_pickle_path, 'rb') as handle:
        tokenizer, model = pickle.load(handle)

    #runcation为True会把过长的输入切掉，从而保证所有的句子都是相同长度的，return_tensors=”pt”表示返回的是PyTorch的Tensor
    # 如果使用TensorFlow则需要设置return_tensors=”tf”。
    inputs = tokenizer(text=text_input, add_special_tokens=True, truncation=True, padding=True, return_tensors='pt')
    #print(inputs)
    #{'input_ids': [101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 19081, 3075, 1012, 102],
    # 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

    tokens = [tokenizer.decode([token_id]) for token_id in inputs['input_ids'][0]]
    n_tokens = len(inputs['input_ids'][0])

    # Forward pass, calculate logit predictions.
    # This will return the logits rather than the loss because we have
    # not provided labels.
    # token_type_ids is the same as the "segment ids", which
    # differentiates sentence 1 and 2 in 2-sentence tasks.
    # The documentation for this `model` function is here:
    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
    # outputs = model(**inputs)

    with torch.no_grad():
        outputs = model.forward(**inputs, output_attentions=True, return_dict=True)

    # Try to free up any memory.
    try:
        # Delete previously created object.
        del inputs
    except ValueError:
        # Print message if not successful.
        print('Not able to free memory.')
        sys.stdout.flush()
    gc.collect()

    #outputs《——model《———model_pickle_path
    logits = outputs['logits'].detach().cpu().numpy()
    attentions = outputs['attentions']

    # Make sure attentions scores are good format in order to get attentions.
    try:
        # Attention is correct format.
        attentions = attentions[-1].squeeze()[0][0]
        # Dobule check it has same length. If not just use None values.
        attentions = attentions.detach().numpy() if len(attentions) == n_tokens else [None] * n_tokens
    except:
        # If something goes wrong add None values.
        attentions = [None] * n_tokens
    # The call to `model` always returns a tuple, so we need to pull the
    # loss value out of the tuple along with the logits. We will use logits
    # later to to calculate training accuracy.
    # logits = outputs[0]
    # logits = outputs['logits']

    # Move logits and labels to CPU
    # logits = logits.detach().cpu().numpy()

    # Get probabilities from logits
    # probs = torch.softmax(logits, dim=-1) # using pytroch
    probs = softmax(vector=logits)[0]  # Using custom function. No need to load torch.
    # Make probabilities % 0-100
    probs *= 100
    # Round to 2 decimal places.
    probs = np.around(probs, 2)

    # get predicitons to list
    predict_content = logits.argmax(axis=-1).flatten().tolist()[0]

    # Predicted label
    label = ids_labels.get(predict_content, 'Unknown')

    #在每个lebel里的所占可能百分比
    labels_percents = {lab: prob for lab, prob in zip(ids_labels.values(), probs)}


    return label, labels_percents, attentions, tokens

