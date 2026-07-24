from rouge_score import rouge_scorer
from bert_score import score

import json
import os
import numpy as np

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

def quantitative_metrics(items):
    items = [i for i in items if len(i['pred_note'])]
    pred_notes = [i['pred_note'].strip() for i in items]
    gold_notes = [i['gold_note'] for i in items]
    
    assert len(pred_notes) == len(gold_notes), print('Number of pred notes and gold notes are not same')
    
    rouge_scores = {'rouge1': {'recall': [], 'precision': [], 'f1': []}, 'rougeL': {'recall': [], 'precision': [], 'f1': []}}
    bert_scores = {'recall': [], 'precision': [], 'f1': []}
    
    # ROUGE--report F1
    for i in range(len(pred_notes)):
        _score = scorer.score(pred_notes[i], gold_notes[i])
        for rouge_type in rouge_scores:
            rouge_scores[rouge_type]['recall'].append(_score[rouge_type].recall)
            rouge_scores[rouge_type]['precision'].append(_score[rouge_type].precision)
            rouge_scores[rouge_type]['f1'].append(_score[rouge_type].fmeasure)
            
    for rouge_type in rouge_scores:
        print(f"ROUGE {rouge_type} recall: {np.mean(rouge_scores[rouge_type]['recall'])}")
        print(f"ROUGE {rouge_type} precision: {np.mean(rouge_scores[rouge_type]['precision'])}")
        print(f"ROUGE {rouge_type} f1: {np.mean(rouge_scores[rouge_type]['f1'])}")

    # BERTScore--report F1
    _score = score(pred_notes, gold_notes, lang='en') # return is precision, recall, and f1
    bert_scores['precision'].append(_score[1])
    bert_scores['recall'].append(_score[1])
    bert_scores['f1'].append(_score[2])
    for key in bert_scores:
        print(f"BERTScore {key}: {np.mean(bert_scores[key])}")
    return

if __name__ == '__main__':
    path = 'metrics_test/soap_prompts'
    files = os.listdir(path)
    for filename in files:
        if filename.endswith('.json'):
            print("#"*5)
            print(f"Calculating for {filename}...")
            print("#"*5)
            
            with open(f'{path}/{filename}', 'rb') as fp:
                data = json.load(fp)
            quantitative_metrics(data)