import json

def load_json(file_path):
    """JSON 파일을 로드하는 함수"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def create_translation_dict(hatespeech_file, non_hatespeech_file):
    """두 JSON 파일을 기반으로 표준어-사투리 변환 딕셔너리를 생성"""
    translation_dict = {}

    # Hatespeech 파일 로드
    hatespeech_data = load_json(hatespeech_file)
    for pair in hatespeech_data:
        translation_dict[pair["standard"]] = pair["dialect"]

    # Non-hatespeech 파일 로드
    non_hatespeech_data = load_json(non_hatespeech_file)
    for pair in non_hatespeech_data:
        if "standard" in pair:  # non-hatespeech 파일에 standard가 있을 경우
            translation_dict[pair["standard"]] = pair["dialect"]

    return translation_dict

def translate_to_dialect(sentence, translation_dict):
    """표준어 문장을 사투리 문장으로 변환"""
    words = sentence.split()
    translated_words = [translation_dict.get(word, word) for word in words]
    return ' '.join(translated_words)

if __name__ == "__main__":
    # JSON 파일 경로
    hatespeech_file = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/gs_hatespeech/hatespeech_pairs_human_annotate.json"
    non_hatespeech_file = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/gs_hatespeech/non-hatespeech_pairs.json"

    # 변환 딕셔너리 생성
    translation_dict = create_translation_dict(hatespeech_file, non_hatespeech_file)

    # 표준어 문장 입력
    standard_sentence = '짱깨는 그입닥쳐라..니들입에 묻은 똥이나 처리하지..'
    # 사투리로 변환
    dialect_sentence = translate_to_dialect(standard_sentence, translation_dict)
    print('표준어 문장:', standard_sentence)
    print("사투리 문장:", dialect_sentence)