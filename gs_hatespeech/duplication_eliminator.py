import json

# 파일 읽기
with open('all_pairs.json', 'r', encoding='utf-8') as f:
    all_pairs = json.load(f)

with open('hatespeech_pairs_human_annotate.json', 'r', encoding='utf-8') as f:
    annotated_pairs = json.load(f)

# annotated_pairs의 standard와 dialect 값을 set으로 저장
annotated_set = {(pair.get('standard', ''), pair.get('dialect', '')) for pair in annotated_pairs}

# all_pairs에서 annotated_pairs에 없는 쌍만 필터링
filtered_pairs = [
    pair for pair in all_pairs
    if (pair.get('standard', ''), pair.get('dialect', '')) not in annotated_set
]

# 결과를 non-hs_pairs.json 파일로 저장
with open('non-hs_pairs.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_pairs, f, ensure_ascii=False, indent=4)

# 제거 후 남은 pair 수 출력
print(f"제거 후 남은 pair 수: {len(filtered_pairs)}")