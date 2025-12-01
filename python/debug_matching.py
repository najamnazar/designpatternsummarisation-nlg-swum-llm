import sys
sys.path.insert(0, 'python')
from evaluate_summaries import load_human_summaries, load_generated_summaries, normalize_project_path
from pathlib import Path

human = load_human_summaries(Path('input/DPS_Human_Summaries.csv'))
nlg = load_generated_summaries(Path('output/summary-output/dps_nlg.csv'), 'Summary')
llm = load_generated_summaries(Path('output/summary-output/llm_summaries.csv'), 'LLM Summary')

human['key'] = human['project'].apply(normalize_project_path) + '::' + human['filename']
nlg['key'] = nlg['project'].apply(normalize_project_path) + '::' + nlg['filename']
llm['key'] = llm['project'].apply(normalize_project_path) + '::' + llm['filename']

print(f'Human unique keys: {human["key"].nunique()}')
print(f'Human total: {len(human)}')
print(f'NLG unique keys: {nlg["key"].nunique()}')
print(f'NLG total: {len(nlg)}')
print(f'LLM unique keys: {llm["key"].nunique()}')
print(f'LLM total: {len(llm)}')

print(f'\nMatching keys (NLG): {len(set(human["key"]) & set(nlg["key"]))}')
print(f'Matching keys (LLM): {len(set(human["key"]) & set(llm["key"]))}')

print(f'\nHuman keys not in NLG (first 10):')
missing_nlg = set(human['key']) - set(nlg['key'])
for k in list(missing_nlg)[:10]:
    print(f'  - {k}')

print(f'\nHuman keys not in LLM (first 10):')
missing_llm = set(human['key']) - set(llm['key'])
for k in list(missing_llm)[:10]:
    print(f'  - {k}')

print(f'\nSample human keys (first 5):')
for k in human['key'].head():
    print(f'  - {k}')

print(f'\nSample NLG keys (first 5):')
for k in nlg['key'].head():
    print(f'  - {k}')

print(f'\nSample LLM keys (first 5):')
for k in llm['key'].head():
    print(f'  - {k}')
