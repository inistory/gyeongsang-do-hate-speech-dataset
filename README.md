# Gyeongsang-do-hate-speech-dataset


## 1. 데이터셋 생성

### 1-1. gs_hatespeech 경로
여기서는 non-hs_pairs.json이랑 hs_human_annotate.json만 보면 됨!

1. 경상도 발화데이터에서 쌍 데이터를 중복없이 {표준어:사투리}쌍 생성
-> all.pairs.json (30358쌍)
```
pair_generator.py
```

2. hs_human_annotate.json (99쌍)
hs_pairs.json : regex로 all.pairs.json에서 욕설만 찾은 것
hs_pairs_100.json : chatgpt로 hs_pairs.json를 augmentation한거
hs_human_annotate.json: augmentation한 것에서 사람이 확인하고 수정한 것

3. non-hs_pairs.json (30358쌍)
hs_human_annotate.json에 있는것을 all_pairs에서 제거한 버전


### 1-2. korean_hatespeech 경로
KOLD/data 경로에 데이터 저장

1. 제공되는 KOLD데이터셋 : kold_v1.json (40429개)

2. gs_kold.json 생성(28989개)
```
python create_dataset.py
```
* hs_pairs_human_annotate.json, non-hs_pairs.json과 토큰 매칭 시켜서 최종데이터만들기(40429개)
* 중복제거(40223개)
* standard랑 dialect가 동일한 데이터 제거 (28989개)
* 한 단어, 두 단어 있는 것들 중 hate 개수 추가 (진행중)
* hate speech위치, hate 개수 추가(진행중)
* OFF외에 외에 커리큘럼러닝에 활용할 수 있는 라벨을 데이터에 추가(진행중)
