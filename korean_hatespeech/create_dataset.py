import json

############1. hs_pairs_human_annotate.json, non-hs_pairs.json과 토큰 매칭 시켜서 최종데이터만들기
input_file = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/korean_hatespeech/KOLD/data/kold_v1.json"
output_file = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/korean_hatespeech/KOLD/data/gs_kold.json"
hatespeech_pairs_file = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/gs_hatespeech/hs_pairs_human_annotate.json"
non_hatespeech_pairs_file = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/gs_hatespeech/non-hs_pairs.json"

# JSON 데이터 읽기
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 변환 딕셔너리 생성 함수
def create_translation_dict(hatespeech_file, non_hatespeech_file):
    translation_dict = {}

    # hatespeech_pairs_file 로드
    hatespeech_data = load_json(hatespeech_file)
    for pair in hatespeech_data:
        if pair["standard"] not in translation_dict:  # 이미 존재하는 키는 무시
            translation_dict[pair["standard"]] = pair["dialect"]

    # non_hatespeech_pairs_file 로드
    non_hatespeech_data = load_json(non_hatespeech_file)
    for pair in non_hatespeech_data:
        if "standard" in pair and "dialect" in pair:  # standard와 dialect가 모두 존재하는 경우
            if pair["standard"] not in translation_dict:  # 이미 존재하는 키는 무시
                translation_dict[pair["standard"]] = pair["dialect"]
        elif "dialect" in pair:  # dialect만 존재하는 경우
            if pair["dialect"] not in translation_dict:  # 이미 존재하는 키는 무시
                translation_dict[pair["dialect"]] = pair["dialect"]

    return translation_dict

# 표준어 문장을 사투리 문장으로 변환하는 함수
def convert_to_dialect(standard_sentence, translation_dict):
    tokens = standard_sentence.split()  # 문장을 토큰 단위로 나눔
    converted_tokens = []
    for token in tokens:
        # translation_dict에서 변환 가능한 토큰을 찾음
        converted_tokens.append(translation_dict.get(token, token))
    return " ".join(converted_tokens)  # 변환된 토큰을 다시 문장으로 합침

# OFF_span을 변환된 dialect에서 추출하는 함수
def extract_off_span_dialect(standard, dialect, off_span):
    if off_span is None:
        return None
    try:
        # standard에서 OFF_span의 위치를 찾음
        start_idx = standard.find(off_span)
        if start_idx == -1:
            return None
        end_idx = start_idx + len(off_span)

        # dialect에서 동일한 위치의 문자열을 추출
        return dialect[start_idx:end_idx]
    except Exception as e:
        print(f"OFF_span 변환 중 오류 발생: {e}")
        return None

# 데이터 로드
data = load_json(input_file)
print(f"kold_v1 데이터 개수: {len(data)}")
# 변환 딕셔너리 생성
translation_dict = create_translation_dict(hatespeech_pairs_file, non_hatespeech_pairs_file)

# 데이터 변환
extracted_data = []
for item in data:
    standard = item.get("comment", "")
    OFF = item.get("OFF", None)
    TGT = item.get("TGT", None)
    OFF_span = item.get("OFF_span", None)

    # 표준어 문장을 사투리 문장으로 변환
    dialect = convert_to_dialect(standard, translation_dict)

    # OFF_span은 변환된 dialect에서 추출
    off_span_dialect = extract_off_span_dialect(standard, dialect, OFF_span)

    # 2. standard와 dialect가 동일하지 않은 경우만 추가
    if standard.replace(" ", "") != dialect.replace(" ", ""):
        extracted_data.append({
            "standard": standard,
            "dialect": dialect,
            "OFF": OFF,
            "TGT": TGT,
            "OFF_span": OFF_span,
            "OFF_span_dialect": off_span_dialect
        })

###############3. 중복 제거
unique_data = {json.dumps(item, ensure_ascii=False): item for item in extracted_data}.values()
extracted_data = list(unique_data)

# 변환된 데이터의 총 개수 출력
print(f"최종 변환된 데이터 개수: {len(extracted_data)}")

# 결과를 새로운 JSON 파일로 저장
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(extracted_data, f, ensure_ascii=False, indent=4)

print(f"변환된 데이터가 {output_file}에 저장되었습니다.")

# standard의 길이가 짧은 순으로 정렬
extracted_data.sort(key=lambda x: len(x["standard"]))

# 변환된 데이터의 총 개수 출력
print(f"최종 변환된 데이터 개수: {len(extracted_data)}")

# 결과를 새로운 JSON 파일로 저장
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(extracted_data, f, ensure_ascii=False, indent=4)

print(f"변환된 데이터가 {output_file}에 저장되었습니다.")