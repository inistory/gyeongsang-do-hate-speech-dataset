import os
import json

def load_json_files_from_directory(directory_path):
    """지정된 디렉토리에서 모든 JSON 파일을 읽어와 데이터를 반환합니다."""
    data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    data.extend(json.load(file))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file {file_path}: {e}")
    return data

def remove_duplicates(data):
    """리스트에서 중복된 항목을 제거합니다."""
    return list({json.dumps(item, ensure_ascii=False) for item in data})

def save_to_json_file(data, output_path):
    """데이터를 JSON 파일로 저장합니다."""
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def main():
    train_dir = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/gs_hatespeech/train_preprocess2"
    valid_dir = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/gs_hatespeech/valid_preprocess2"
    output_file = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/gs_hatespeech/all_pairs.json"

    # 두 디렉토리에서 데이터를 읽어옵니다.
    train_data = load_json_files_from_directory(train_dir)
    valid_data = load_json_files_from_directory(valid_dir)

    # 데이터를 합치고 중복을 제거합니다.
    combined_data = train_data + valid_data
    unique_data = remove_duplicates(combined_data)

    # 문자열 데이터를 딕셔너리로 변환
    dict_data = [json.loads(item) for item in unique_data]

    # 결과를 JSON 파일로 저장합니다.
    save_to_json_file(dict_data, output_file)

    # pair 개수 출력
    print(f"딕셔너리 형태로 데이터가 {output_file}에 저장되었습니다.")
    print(f"총 pair 개수: {len(dict_data)}")

if __name__ == "__main__":
    main()