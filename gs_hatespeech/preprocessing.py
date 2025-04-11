import json
import os

# JSON 파일이 있는 디렉터리 경로
input_dir = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/Validation/[라벨]경상도_학습데이터_2"
output_dir1 = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/valid_preprocess"
output_dir2 = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/valid_preprocess2"

# 출력 디렉터리가 없으면 생성
os.makedirs(output_dir1, exist_ok=True)
os.makedirs(output_dir2, exist_ok=True)

# 디렉터리 내 모든 JSON 파일 처리
for file_name in os.listdir(input_dir):
    if file_name.endswith(".json"):
        file_path = os.path.join(input_dir, file_name)
        
        # JSON 파일 로드
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # 발화 데이터 추출
        utterances = data.get("utterance", [])

        # 전처리 결과 저장
        processed_data1 = []  # preprocess 폴더에 저장할 데이터
        processed_data2 = []  # preprocess2 폴더에 저장할 데이터

        # 발화 번호와 어절 단위 표준어:방언 쌍 추출
        for utterance in utterances:
            utterance_id = utterance.get("id", "")
            eojeol_list = utterance.get("eojeolList", [])
            
            # 발화 번호와 말 번호 분리
            if utterance_id:
                utterance_parts = utterance_id.split(".")
                if len(utterance_parts) >= 4:
                    utterance_number = utterance_parts[2]
                    sentence_number = utterance_parts[3]
                else:
                    continue  # ID 형식이 맞지 않으면 건너뜀
            else:
                continue  # ID가 없으면 건너뜀

            # 어절 단위 표준어:방언 쌍 생성
            for eojeol in eojeol_list:
                standard = eojeol.get("standard", "")
                dialect = eojeol.get("eojeol", "")
                is_dialect = eojeol.get("isDialect", False)
                
                # 방언인 경우만 추가
                if is_dialect:
                    # preprocess 폴더에 저장할 데이터
                    processed_data1.append({
                        "utterance_number": utterance_number,
                        "sentence_number": sentence_number,
                        "standard": standard,
                        "dialect": dialect
                    })
                    # preprocess2 폴더에 저장할 데이터
                    processed_data2.append({
                        "standard": standard,
                        "dialect": dialect
                    })

        # 결과를 preprocess 폴더에 저장
        output_file_path1 = os.path.join(output_dir1, f"processed_{file_name}")
        with open(output_file_path1, "w", encoding="utf-8") as output_file1:
            json.dump(processed_data1, output_file1, ensure_ascii=False, indent=4)

        # 결과를 preprocess2 폴더에 저장
        output_file_path2 = os.path.join(output_dir2, f"processed2_{file_name}")
        with open(output_file_path2, "w", encoding="utf-8") as output_file2:
            json.dump(processed_data2, output_file2, ensure_ascii=False, indent=4)

        print(f"{file_name}의 전처리 결과가 {output_file_path1}와 {output_file_path2}에 저장되었습니다.")