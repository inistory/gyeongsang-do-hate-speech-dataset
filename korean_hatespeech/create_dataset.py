import json

# 파일 경로 설정
input_file = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/korean_hatespeech/KOLD/data/kold_v1.json"
output_file = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/korean_hatespeech/KOLD/data/gs_kold.json"
hatespeech_pairs_file = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/gs_hatespeech/hatespeech_pairs_human_annotate.json"
non_hatespeech_pairs_file = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/gs_hatespeech/non-hatespeech_pairs.json"

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
        translation_dict[pair["standard"]] = pair["dialect"]

    # non_hatespeech_pairs_file 로드
    non_hatespeech_data = load_json(non_hatespeech_file)
    for pair in non_hatespeech_data:
        if "standard" in pair and "dialect" in pair:  # standard와 dialect가 모두 존재하는 경우
            translation_dict[pair["standard"]] = pair["dialect"]
        elif "dialect" in pair:  # dialect만 존재하는 경우
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

# 데이터 로드
data = load_json(input_file)

# 변환 딕셔너리 생성
translation_dict = create_translation_dict(hatespeech_pairs_file, non_hatespeech_pairs_file)

# 데이터 변환
extracted_data = []
for item in data:
    standard = item.get("comment", "")
    OFF = item.get("OFF", None)

    # 표준어 문장을 사투리 문장으로 변환
    dialect = convert_to_dialect(standard, translation_dict)

    extracted_data.append({
        "standard": standard,
        "dialect": dialect,
        "OFF": OFF
    })

# 결과를 새로운 JSON 파일로 저장
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(extracted_data, f, ensure_ascii=False, indent=4)

print(f"변환된 데이터가 {output_file}에 저장되었습니다.")