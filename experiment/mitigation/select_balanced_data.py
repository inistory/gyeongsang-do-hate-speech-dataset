import json
from collections import defaultdict
import random

def count_ones(labels):
    return sum(1 for label in labels if label == 1)

def select_balanced_data(input_file, output_file, num_samples=100):
    # 데이터 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # TGT와 labels 1의 개수에 따라 데이터 분류
    tgt_groups = defaultdict(list)
    for item in data:
        tgt = item['TGT']
        ones_count = count_ones(item['labels'])
        
        if ones_count <= 13:
            group = 'low'
        elif ones_count <= 27:
            group = 'medium'
        else:
            group = 'high'
            
        tgt_groups[(tgt, group)].append(item)
    
    # 각 그룹별로 균형있게 선택
    selected_data = []
    samples_per_group = num_samples // (len(set(tgt for tgt, _ in tgt_groups.keys())) * 3)
    
    for (tgt, group), items in tgt_groups.items():
        if len(items) >= samples_per_group:
            selected = random.sample(items, samples_per_group)
        else:
            selected = items
        selected_data.extend(selected)
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(selected_data, f, ensure_ascii=False, indent=2)
    
    # 선택된 데이터 통계 출력
    print(f"선택된 데이터 수: {len(selected_data)}")
    for (tgt, group), items in tgt_groups.items():
        print(f"TGT: {tgt}, Group: {group}, 선택된 수: {len(items)}")

if __name__ == "__main__":
    input_file = "experiment/gs_kold_test.json"
    output_file = "experiment/mitigation/selected_data.json"
    select_balanced_data(input_file, output_file) 