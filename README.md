# Gyeongsang-do-hate-speech-dataset


## 1. 데이터셋 생성

> ### 1-1. gs_hatespeech 경로
여기서는 non-hs_pairs.json이랑 hs_human_annotate.json만 보면 됨!

1. 경상도 발화데이터에서 쌍 데이터를 중복없이 {표준어:사투리}쌍 생성
* all.pairs.json (30358쌍)
```
pair_generator.py
```

2. hs_human_annotate.json (99쌍)
* hs_pairs.json : regex로 all.pairs.json에서 욕설만 찾은 것
* hs_pairs_100.json : chatgpt로 hs_pairs.json를 augmentation한거
* hs_human_annotate.json: augmentation한 것에서 사람이 확인하고 수정한 것

3. non-hs_pairs.json (30358쌍)
* hs_human_annotate.json에 있는것을 all_pairs에서 제거한 버전


> ### 1-2. korean_hatespeech 경로

1. 제공되는 KOLD데이터셋 : kold_v1.json (40429개)

2. gs_kold.json 생성(28989개)
```
python create_dataset.py
```
```
{
    "standard": "표준어문장",
    "dialect": "경상도사투리문장",
    "OFF": "true/false(hate여부)", 
    "TGT": [level1=OFF, 
            level2={UNT, IND, OTH},
            level3={GRP}
           ],
    "OFF_span": "공격적인 표현이나 타깃 표현이 실제로 등장하는 위치"
}
```
* hs_pairs_human_annotate.json, non-hs_pairs.json과 토큰 매칭 시켜서 최종데이터만들기(40429개)
* 중복제거(40223개)
* standard랑 dialect가 동일한 데이터 제거 (28989개)
* 한 단어, 두 단어 있는 것들 중 hate 개수 추가 (진행중)
* hate speech위치추가(OFF_span)
* 커리큘럼 러닝을 위한 TGT 추가: 
  * null: not hate
  * UNT (Untargeted): 특정한 대상 없이 막연히 욕설이나 비속어만 포함된 경우 (예: "개짜증나")
  * IND (Individual): 특정 개인(유명인, 특정인을 지칭하는 표현 등)을 겨냥한 공격 (사이버불링 포함)
  * OTH (Other): 개인도 집단도 아닌 조직/회사/사건 등을 타깃으로 한 경우 (예: "이 정부는 답이 없다")
  * GRP (Group): 인종, 성별, 정치 성향 등 사회적 집단을 겨냥한 경우 (예: "여자는 운전 못 해")
