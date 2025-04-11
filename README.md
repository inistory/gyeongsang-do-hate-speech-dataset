# Gyeongsang-do-hate-speech-dataset


1. 데이터셋 생성

1-1. gs_hatespeech 경로
여기서는 non-hs_pairs.json이랑 hs_human_annotate.json만 보면 됨!

```
#1.경상도 발화데이터에서 쌍 데이터를 중복없이 {표준어:사투리}쌍 생성
pair_generator.py -> all.pairs.json (30358쌍)


#2.hs_human_annotate.json (99쌍)
hs_pairs.json : regex로 all.pairs.json에서 욕설만 찾은 것
hs_pairs_100.json : chatgpt로 hs_pairs.json를 augmentation한거
hs_human_annotate.json: augmentation한 것에서 사람이 확인하고 수정한 것

#3. non-hs_pairs.json (30358쌍)
hs_human_annotate.json에 있는것을 all_pairs에서 제거한 버전

```


1-1. korean_hatespeech 경로
```
KOLD/data

kold_v1.json: KOLD기본 데이터
gs_kold.json:
```