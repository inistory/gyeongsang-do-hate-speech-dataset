import json

# 파일 경로
input_file = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/korean_hatespeech/KOLD/data/comments_and_off.json"
output_file = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/korean_hatespeech/KOLD/data/comments_and_off_with_standard.json"
unmatched_file = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/korean_hatespeech/KOLD/data/unmatched_tokens.json"
hatespeech_file = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/gs_hatespeech/hatespeech_pairs_human_annotate.json"
non_hatespeech_file = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/gs_hatespeech/non-hatespeech_pairs.json"

# JSON 파일 읽기
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(hatespeech_file, "r", encoding="utf-8") as f:
    hatespeech_pairs = json.load(f)

with open(non_hatespeech_file, "r", encoding="utf-8") as f:
    non_hatespeech_pairs = json.load(f)

# hatespeech와 non-hatespeech의 토큰 쌍 개수 출력
print(f"hatespeech_pairs_human_annotate.json의 토큰 쌍 개수: {len(hatespeech_pairs)}")
print(f"non-hatespeech_pairs.json의 토큰 쌍 개수: {len(non_hatespeech_pairs)}")

# 1차적으로 데이터를 처리하여 고유 토큰 추출
unique_tokens = set()
for entry in data:
    if "comment" in entry:
        tokens = entry["comment"].split()  # comment를 띄어쓰기 단위로 자름
        unique_tokens.update(tokens)

# 1차 데이터 개수 출력
print(f"1차적으로 추출된 고유 토큰 개수: {len(unique_tokens)}")

# hatespeech와 non-hatespeech의 standard와 dialect 매핑 생성
dialect_mapping = {}
for pair in hatespeech_pairs:
    dialect_mapping[pair["standard"]] = pair["dialect"]

for pair in non_hatespeech_pairs:
    dialect_mapping[pair["dialect"]] = pair["dialect"]

# 2차적으로 dialect 값을 추가
processed_data = []
unmatched_tokens = []
mapped_count = 0
unmapped_count = 0

for token in sorted(unique_tokens):
    dialect = dialect_mapping.get(token, "")  # 매핑된 값이 없으면 빈 문자열
    if dialect:
        mapped_count += 1
        processed_data.append({"standard": token, "dialect": dialect})
    else:
        unmapped_count += 1
        unmatched_tokens.append({"standard": token, "dialect": ""})

# 매핑 결과 출력
print(f"매핑 성공한 토큰 수: {mapped_count}")
print(f"매핑되지 않은 토큰 수: {unmapped_count}")

# 결과를 새로운 JSON 파일로 저장
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=4)

# 매칭되지 않은 토큰을 별도로 저장
with open(unmatched_file, "w", encoding="utf-8") as f:
    json.dump(unmatched_tokens, f, ensure_ascii=False, indent=4)

print(f"Processed data with dialects has been saved to {output_file}")
print(f"Unmatched tokens have been saved to {unmatched_file}")

import re

# 추가 파일 경로
preprocessed_file = "/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/korean_hatespeech/KOLD/data/unmatched_tokens_preprocessed.json"

# 정규식 패턴
profanity_regex = re.compile(
    r"[시씨씪슈쓔쉬쉽쒸쓉](?:[0-9]*|[0-9]+ *)[바발벌빠빡빨뻘파팔펄]|[섊좆좇졷좄좃좉졽썅춍봊]|"
    r"[ㅈ조][0-9]*까|ㅅㅣㅂㅏㄹ?|ㅂ[0-9]*ㅅ|[ㅄᄲᇪᄺᄡᄣᄦᇠ]|[ㅅㅆᄴ][0-9]*[ㄲㅅㅆᄴㅂ]|"
    r"[존좉좇][0-9 ]*나|[자보][0-9]+지|보빨|[봊봋봇봈볻봁봍] *[빨이]|[후훚훐훛훋훗훘훟훝훑][장앙]|"
    r"[엠앰]창|애[미비]|애자|[가-탏탏-힣]색기|(?:[샊샛세쉐쉑쉨쉒객갞갟갯갰갴겍겎겏겤곅곆곇곗곘곜걕걖걗걧걨걬] *[끼키퀴])|"
    r"새 *[키퀴]|[병븅][0-9]*[신딱딲]|미친[가-닣닥-힣]|[믿밑]힌|[염옘][0-9]*병|[샊샛샜샠섹섺셋셌셐셱솃솄솈섁섂섓섔섘]기|"
    r"[섹섺섻쎅쎆쎇쎽쎾쎿섁섂섃썍썎썏][스쓰]|[지야][0-9]*랄|니[애에]미|갈[0-9]*보[^가-힣]|[뻐뻑뻒뻙뻨][0-9]*[뀨큐킹낑)|"
    r"꼬[0-9]*추|곧[0-9]*휴|[가-힣]슬아치|자[0-9]*박꼼|빨통|[사싸](?:이코|가지|[0-9]*까시)|육[0-9]*시[랄럴]|"
    r"육[0-9]*실[알얼할헐]|즐[^가-힣]|찌[0-9]*(?:질이|랭이)|찐[0-9]*따|찐[0-9]*찌버거|창[녀놈]|[가-힣]{2,}충[^가-힣]|"
    r"[가-힣]{2,}츙|부녀자|화냥년|환[양향]년|호[0-9]*[구모]|조[선센][징]|조센|[쪼쪽쪾](?:[발빨]이|[바빠]리)|"
    r"盧|무현|찌끄[레래]기|(?:하악){2,}|하[앍앜]|[낭당랑앙항남담람암함][ ]?[가-힣]+[띠찌]|느[금급]마|文在|在寅|"
    r"(?<=[^\n])[家哥]|속냐|[tT]l[qQ]kf|Wls|[ㅂ]신|[ㅅ]발|[ㅈ]밥"
)

# 욕설 필터링
filtered_tokens = [
    token for token in unmatched_tokens if profanity_regex.search(token["standard"])
]

# 필터링된 결과 저장
with open(preprocessed_file, "w", encoding="utf-8") as f:
    json.dump(filtered_tokens, f, ensure_ascii=False, indent=4)

print(f"Filtered tokens with profanity have been saved to {preprocessed_file}")
print(f"Number of filtered tokens: {len(filtered_tokens)}")