import json

def mask_offensive_terms(input_file, output_file):
    # 데이터 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    masked_data = []
    skipped_count = 0
    for item in data:
        dialect = item['dialect']
        off_span = item['OFF_span_dialect']
        
        # OFF_span_dialect가 None인 경우 건너뛰기
        if off_span is None:
            skipped_count += 1
            continue
        
        # 비속어 표현을 [MASK]로 대체
        masked_dialect = dialect.replace(off_span, '[MASK]')
        
        # 기존 데이터의 모든 필드를 복사하고 마스킹 관련 필드 추가
        masked_item = item.copy()  # 기존 데이터의 모든 필드 복사
        masked_item.update({
            'original_dialect': dialect,
            'masked_dialect': masked_dialect,
            'offensive_span': off_span
        })
        masked_data.append(masked_item)
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(masked_data, f, ensure_ascii=False, indent=2)
    
    print(f"마스킹된 데이터 수: {len(masked_data)}")
    print(f"건너뛴 데이터 수: {skipped_count}")

if __name__ == "__main__":
    input_file = "experiment/mitigation/selected_data.json"
    output_file = "experiment/mitigation/masked_data.json"
    mask_offensive_terms(input_file, output_file) 