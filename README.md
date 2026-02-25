# karpathy microgpt 학습 기록

- 📌 원본 코드: [microgpt.py (Karpathy Gist)](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- 📖 해설 블로그: [karpathy.github.io/2026/02/12/microgpt](https://karpathy.github.io/2026/02/12/microgpt/)

---



# 1. ch1_data_and_tokenizer.py


## 1.1 코드 설명

[ch1_data_and_tokenizer.py](study/ch1_data_and_tokenizer.py)

### 1.1.1 데이터셋
- 30000여개의 이름 데이터셋을 기반으로 학습
- 목적은 그럴듯한 이름을 생성(**환각**) 하는것

### 1.1.2 토크나이저
- 내부적으로 신경망은 문자가 아닌 숫자로 작동한다.
- 따라서 텍스트를 정수 토큰 ID의 시퀀스로 변환하고 다시 되돌리는 방법이 필요하다.
    - ex) a=0, b=1, ..., z=25, BOS=26 으로 ID 부여
    - BOS 는 문자의 시작과 끝을 나타내는 특수 토큰
    > [!IMPORTANT]
    > **각 고유 문자에 숫자를 할당, 정수 자체에는 아무 의미가 없다!**
- 학습 시 "emma" → [26, 4, 12, 12, 0, 26] → 5개의 (입력, 정답) 쌍
    - 입력(input): [26, 4, 12, 12, 0]
    - 정답(target): [4, 12, 12, 0, 26] -> 다음 토큰 예측


