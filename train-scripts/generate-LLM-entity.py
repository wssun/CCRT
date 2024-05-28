# Import the os package
import os
import pandas as pd
import argparse
import torch
import pathlib
import requests
import random
import pickle
import openai
import csv

class individual:
    def __init__(self, id, concepts = None, score = 0.0):
        if concepts is None:
            self.concepts = []
        else:
            self.concepts = concepts
        self.id = id
        self.score = score



def openai_reply(concepts, question):
    openai.api_key = "xx"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",  # gpt-3.5-turbo-0301
        messages=[
            {"role": "user", "content": f"{question}{concepts}"}
        ],
        temperature=0.5,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    # print(response)
    return response.choices[0].message.content



if __name__=='__main__':
    parser = argparse.ArgumentParser(
        prog='generat LLM entity',)
    parser.add_argument('--pkl_name', help='after run generate-entity.py', type=str, required=True)
    args = parser.parse_args()
    pkl_name = args.pkl_name
    with open(f'pkl/{pkl_name}.pkl', 'rb') as f:
        loaded_individuals = pickle.load(f)

    question = "I will give you a list of multiple strings, each describing a different concept, and ask you to build the most concise text that roughly contains these concepts, which can be used as a prompt to generate an image, but only as long as it describes the content of the picture.. The list is as follows:"
    data_rows = []

    for i, individual in enumerate(loaded_individuals, 1):
        ids = individual.id
        concepts = individual.concepts
        new_concepts = [item.split(':')[0].strip() for item in concepts]
        new_list = [random.choice(item.split(',')) for item in new_concepts]
        reply = openai_reply(new_list,question)
        data_rows.append((str(i),reply))

    with open(f'../data/entity-{pkl_name}.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for row in data_rows:
            writer.writerow(row)







