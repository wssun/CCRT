import pandas as pd
import os

directory = 'humanTest\\result\\'
files = [f for f in os.listdir(directory) if f.startswith('eval_') and f.endswith('.xlsx')]

all_averages_dicts = []

for file in files:
    filepath = os.path.join(directory, file)

    df = pd.read_excel(filepath, usecols=[2])
    score_list = df.iloc[:, 0].tolist()

    scores_2d = [score_list[i:i + 9] for i in range(0, len(score_list), 9)]

    averages_dicts = []

    for row in scores_2d:
        textImgAlign = round(sum(row[:3]) / 3 if len(row) >= 3 else 0, 5)
        preserveOtherStyle = round(sum(row[3:6]) / 3 if len(row) >= 6 else 0, 5)
        eraseStyle =  round(sum(row[6:]) / 3 if len(row) >= 6 else 0, 5)

        averages_dict = {
            'textImgAlign': textImgAlign,
            'preserveOtherStyle': preserveOtherStyle,
            'eraseStyle': eraseStyle
        }


        averages_dicts.append(averages_dict)
    all_averages_dicts.append(averages_dicts)

final_averages_dicts = []
for i in range(len(all_averages_dicts[0]) if all_averages_dicts else 0):
    avg_dict = {
        'textImgAlign': round(sum(d[i]['textImgAlign'] for d in all_averages_dicts if i < len(d)) / len(
            [d for d in all_averages_dicts if i < len(d)]), 3) ,
        'preserveOtherStyle': round(sum(d[i]['preserveOtherStyle'] for d in all_averages_dicts if i < len(d)) / len(
            [d for d in all_averages_dicts if i < len(d)]), 3) ,
        'eraseStyle': round(sum(d[i]['eraseStyle'] for d in all_averages_dicts if i < len(d)) / len(
            [d for d in all_averages_dicts if i < len(d)]), 3)
    }
    final_averages_dicts.append(avg_dict)

for avg_dict in final_averages_dicts:
    print(avg_dict)

row_names = ['SD','ESD0', 'UCE0', 'OURS0', 'ESD1', 'UCE1', 'OURS1', 'ESD2', 'UCE2', 'OURS2', 'ESD3', 'UCE3', 'OURS3']
df = pd.DataFrame(final_averages_dicts)
df.index = row_names


df.to_excel(f'{directory}average_results.xlsx', index=True, engine='openpyxl')