"""
=============================================================================
 Chapter 1: Dataset & Tokenizer
 - microgpt.py L1~26에 해당하는 코드
 - explain.md의 Dataset, Tokenizer 섹션 내용을 한국어 주석으로 포함
 - 독립 실행 가능: python study/ch1_data_and_tokenizer.py
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 섹션 1: Dataset (데이터셋)
# ─────────────────────────────────────────────────────────────────────────────
#
# [explain.md - Dataset 섹션 요약]
# 대규모 언어 모델(LLM)의 연료는 텍스트 데이터 스트림이다.
# 프로덕션 애플리케이션에서는 각 문서가 인터넷 웹페이지이지만,
# microgpt에서는 더 간단한 예시로 32,000개의 이름을 사용한다.
#
# 모델의 목표: 데이터의 패턴을 학습하고, 그와 유사한 통계적 패턴을 가진
# 새로운 문서(이름)를 생성하는 것.
#
# ChatGPT 관점에서 보면, 당신과의 대화도 그냥 좀 특이하게 생긴 "문서"일 뿐이다.
# 프롬프트로 문서를 시작하면, 모델의 응답은 단지 통계적 문서 완성에 불과하다.
# ─────────────────────────────────────────────────────────────────────────────

import os
import random
random.seed(42)  # 재현 가능한 결과를 위한 시드 고정

# 디버거(LazyVim 등)에서 cwd가 study/일 때 input.txt를 찾기 위해 프로젝트 루트로 이동
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# --- 원본 코드 (microgpt.py L14~20) ---
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]  # 🔴 BREAKPOINT: docs의 타입, 길이, 첫 몇 개 원소 확인
random.shuffle(docs)

# ─── 디버그 포인트 1: 데이터셋 관찰 ─────────────────────────────────────────
print("=" * 60)
print("📦 [디버그] 데이터셋 관찰")
print("=" * 60)
print(f"  총 문서(이름) 수: {len(docs)}")
print(f"  첫 10개 이름: {docs[:10]}")
print(f"  마지막 5개 이름: {docs[-5:]}")
print(f"  가장 긴 이름: '{max(docs, key=len)}' (길이: {len(max(docs, key=len))})")
print(f"  가장 짧은 이름: '{min(docs, key=len)}' (길이: {len(min(docs, key=len))})")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 섹션 2: Tokenizer (토크나이저)
# ─────────────────────────────────────────────────────────────────────────────
#
# [explain.md - Tokenizer 섹션 요약]
# 신경망은 문자가 아닌 숫자로 작동한다.
# 따라서 텍스트를 정수 토큰 ID의 시퀀스로 변환하고 다시 되돌리는 방법이 필요하다.
#
# 프로덕션 토크나이저(예: GPT-4의 tiktoken)는 효율성을 위해
# 문자 청크 단위로 작동하지만,
# 가장 단순한 토크나이저는 데이터셋의 각 고유 문자에 정수 하나를 할당한다.
#
# 핵심 포인트:
# - 정수 값 자체에는 의미가 없다. 각 토큰은 별개의 이산(discrete) 심볼이다.
#   0, 1, 2 대신 서로 다른 이모지여도 동일하게 작동한다.
# - BOS(Beginning of Sequence) 토큰은 구분자 역할:
#   "새로운 문서가 여기서 시작/끝난다"를 모델에게 알려준다.
# - 학습 시 각 문서는 양쪽에 BOS로 감싸진다:
#   "emma" → [BOS, e, m, m, a, BOS]
# - 최종 어휘 크기: 27 (26개 소문자 a-z + 1개 BOS 토큰)
# ─────────────────────────────────────────────────────────────────────────────

# --- 원본 코드 (microgpt.py L22~26) ---
uchars = sorted(set(''.join(docs)))  # 🔴 BREAKPOINT: set(''.join(docs))가 뭘 반환하는지, 정렬 후 결과 확인
BOS = len(uchars)                     # 🔴 BREAKPOINT: BOS가 26인지 확인
vocab_size = len(uchars) + 1          # vocab_size가 27인지 확인

# ─── 디버그 포인트 2: 토크나이저 관찰 ───────────────────────────────────────
print("=" * 60)
print("🔤 [디버그] 토크나이저 관찰")
print("=" * 60)
print(f"  고유 문자(uchars): {uchars}")
print(f"  고유 문자 수: {len(uchars)}")
print(f"  BOS 토큰 ID: {BOS}")
print(f"  vocab_size: {vocab_size}")
print()

# 문자 → 토큰 ID 매핑 테이블 출력
print("  [문자 → 토큰 ID 매핑]")
for i, ch in enumerate(uchars):
    print(f"    '{ch}' → {i}", end="")
    if (i + 1) % 6 == 0:
        print()
print(f"\n    BOS → {BOS}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 섹션 3: "emma"를 직접 토크나이징해보기
# ─────────────────────────────────────────────────────────────────────────────
#
# [explain.md 인용]
# 학습 시 각 문서는 양쪽을 BOS로 감싼다:
# 이름 "emma"는 [BOS, e, m, m, a, BOS]가 된다.
# 모델의 임무는 이전 토큰들이 주어졌을 때 다음 토큰을 예측하는 것이다.
#
# 즉, 학습 데이터에서 (입력, 정답) 쌍은 이렇게 만들어진다:
#   위치 0: 입력=BOS → 정답=e   ("이름이 시작되면, 다음은 e가 올 것")
#   위치 1: 입력=e   → 정답=m   ("e 다음에는 m이 올 것")
#   위치 2: 입력=m   → 정답=m   ("m 다음에는 m이 올 것")
#   위치 3: 입력=m   → 정답=a   ("m 다음에는 a가 올 것")
#   위치 4: 입력=a   → 정답=BOS ("a 다음에는 이름이 끝날 것")
# ─────────────────────────────────────────────────────────────────────────────

word = "emma"

print("=" * 60)
print(f"📝 [실습] '{word}' 토크나이징 과정 따라가기")
print("=" * 60)

# Step 1: 문자 → 토큰 ID 변환
char_tokens = [uchars.index(ch) for ch in word]  # 🔴 BREAKPOINT: "emma"의 각 글자가 어떤 ID로 변환되는지 한 글자씩
print(f"\n  Step 1) 문자별 토큰 변환:")
for ch in word:
    token_id = uchars.index(ch)
    print(f"    '{ch}' → uchars.index('{ch}') = {token_id}")

# Step 2: BOS로 감싸기
tokens = [BOS] + char_tokens + [BOS]  # 🔴 BREAKPOINT: 최종 토큰 시퀀스 [26, 4, 12, 12, 0, 26] 확인
print(f"\n  Step 2) BOS로 감싸기:")
print(f"    [BOS] + [{', '.join(str(t) for t in char_tokens)}] + [BOS]")
print(f"    = {tokens}")

# Step 3: 토큰 → 문자 복원 (디코딩)
print(f"\n  Step 3) 토큰 → 문자 복원:")
decoded = ""
for t in tokens:
    if t == BOS:
        print(f"    토큰 {t:2d} → [BOS]")
    else:
        ch = uchars[t]
        decoded += ch
        print(f"    토큰 {t:2d} → '{ch}'")
print(f"    복원된 단어: '{decoded}'")

# Step 4: 학습 데이터 (입력-정답 쌍) 구성
#   이게 실제 학습 루프에서 쓰이는 방식 (microgpt.py L156~157 참고)
print(f"\n  Step 4) 학습 데이터 (입력→정답) 쌍:")
n = len(tokens) - 1  # 예측할 수 있는 위치의 수
for pos_id in range(n):
    input_token = tokens[pos_id]  # 🔴 BREAKPOINT: 매 반복마다 input_token, target_token 쌍 관찰
    target_token = tokens[pos_id + 1]

    # 사람이 읽을 수 있는 형태로 변환
    input_str = "[BOS]" if input_token == BOS else f"'{uchars[input_token]}'"
    target_str = "[BOS]" if target_token == BOS else f"'{uchars[target_token]}'"

    print(f"    위치 {pos_id}: 입력={input_str:>5s} (id={input_token:2d}) → 정답={target_str:>5s} (id={target_token:2d})")

print()

# ─────────────────────────────────────────────────────────────────────────────
# 섹션 4: 다른 단어로도 실험해보기
# ─────────────────────────────────────────────────────────────────────────────
# 💡 직접 바꿔보세요! 다른 이름으로 어떻게 토크나이징되는지 관찰하기.

print("=" * 60)
print("🧪 [실험] 다양한 이름 토크나이징")
print("=" * 60)

test_words = ["sophia", "mia", "a", "zzz"]
for w in test_words:
    try:
        toks = [BOS] + [uchars.index(ch) for ch in w] + [BOS]
        tok_str = " ".join(str(t) for t in toks)
        print(f"  '{w:10s}' → [{tok_str}]  (토큰 {len(toks)}개, 학습 쌍 {len(toks)-1}개)")
    except ValueError as e:
        print(f"  '{w:10s}' → 오류! 데이터셋에 없는 문자 포함: {e}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# 섹션 5: 프로덕션 토크나이저와의 차이
# ─────────────────────────────────────────────────────────────────────────────
#
# [explain.md - Real stuff 섹션 요약]
# - microgpt: 단일 문자 토크나이저 (vocab_size = 27)
# - 프로덕션(GPT-4 등): BPE(Byte Pair Encoding) 같은 서브워드 토크나이저 사용
#   → 자주 같이 나타나는 문자 시퀀스를 하나의 토큰으로 병합
#   → "the" 같은 흔한 단어 = 하나의 토큰
#   → 드문 단어는 조각으로 분해
#   → 어휘 크기 ~100K 토큰
#   → 위치당 더 많은 내용을 보기 때문에 훨씬 효율적
#
# 예시 비교:
#   microgpt: "hello" → [h, e, l, l, o] → 5 토큰
#   GPT-4:    "hello" → [hello]          → 1 토큰
#
# 알고리즘적으로 동일하지만, 효율성에서 큰 차이!
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("📊 [비교] microgpt vs 프로덕션 토크나이저")
print("=" * 60)
comparison_word = "hello world"
our_tokens = []
for ch in comparison_word:
    if ch in uchars:
        our_tokens.append(f"'{ch}'")
    else:
        our_tokens.append(f"'{ch}'(없음!)")
print(f"  microgpt 토크나이저:  '{comparison_word}' → {our_tokens}")
print(f"  (우리 vocab에는 소문자 a-z만 있어서 공백은 처리 불가)")
print(f"  GPT-4 토크나이저:     '{comparison_word}' → ['hello', ' world'] (약 2토큰)")
print(f"  효율성 차이: {len(comparison_word)}자를 우리는 {len(comparison_word)}토큰, GPT-4는 ~2토큰으로 처리")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 핵심 정리
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("🎯 핵심 정리")
print("=" * 60)
print("""
  1. 데이터셋: 32,000개의 이름이 각각 하나의 "문서"
  2. 토크나이저: 각 고유 문자에 정수 ID를 부여 (a=0, b=1, ..., z=25)
  3. BOS 토큰(id=26): 문서의 시작과 끝을 나타내는 특수 토큰
  4. vocab_size = 27: 26개 문자 + 1개 BOS
  5. 학습 시 "emma" → [26, 4, 12, 12, 0, 26] → 5개의 (입력, 정답) 쌍

  💡 다음 챕터에서 이 토큰들이 신경망에 어떻게 입력되는지 배운다.
""")
