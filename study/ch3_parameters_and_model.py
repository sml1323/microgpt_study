"""
=============================================================================
 Chapter 3: Parameters & Model Architecture (파라미터와 모델 구조)
 - microgpt.py L73~143에 해당하는 코드
 - explain.md의 Parameters, Architecture 섹션 내용을 한국어 주석으로 포함
 - 디버그용 축소 파라미터: n_embd=4, n_head=2, n_layer=1
 - 독립 실행 가능: python study/ch3_parameters_and_model.py
=============================================================================
"""

import os
import math
import random
random.seed(42)

# 디버거(LazyVim 등)에서 cwd가 study/일 때 input.txt를 찾기 위해 프로젝트 루트로 이동
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# ─────────────────────────────────────────────────────────────────────────────
# 사전 준비: Value 클래스 (ch2에서 가져옴, 독립 실행을 위해 포함)
# ─────────────────────────────────────────────────────────────────────────────
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

    def __repr__(self):
        return f"Value(data={self.data:.4f})"

# ─────────────────────────────────────────────────────────────────────────────
# 사전 준비: 토크나이저 (ch1에서 가져옴, 독립 실행을 위해 포함)
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)

uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab_size: {vocab_size}")


# =============================================================================
# 섹션 1: 파라미터 초기화 (microgpt.py L73~89)
# =============================================================================
#
# [explain.md - Parameters 섹션 요약]
# 파라미터는 모델의 지식이다.
# 부동소수점 숫자들(Value로 감싸진)의 큰 집합으로,
# 처음에는 랜덤으로 시작하고 학습 중에 반복적으로 최적화된다.
#
# 각 파라미터는 가우시안 분포에서 뽑은 작은 랜덤 숫자로 초기화된다.
# state_dict는 이름 붙인 행렬들로 구성:
#   - 임베딩 테이블 (wte, wpe)
#   - 어텐션 가중치 (attn_wq, attn_wk, attn_wv, attn_wo)
#   - MLP 가중치 (mlp_fc1, mlp_fc2)
#   - 출력 프로젝션 (lm_head)
#
# 우리의 축소 모델: 약 수백 개 파라미터
# 원본 microgpt: 4,192개 파라미터
# GPT-2: 16억 개, 현대 LLM: 수천억 개
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("🔧 [섹션 1] 파라미터 초기화")
print("=" * 60)

# --- 💡 디버그용 축소 파라미터 ---
# 원본: n_layer=1, n_embd=16, block_size=16, n_head=4
# 축소: n_layer=1, n_embd=4,  block_size=8,  n_head=2
n_layer = 1       # 트랜스포머 레이어 수 (깊이)
n_embd = 4        # 임베딩 차원 (너비) ← 원본 16에서 4로 축소
block_size = 8    # 최대 시퀀스 길이 ← 원본 16에서 8로 축소
n_head = 2        # 어텐션 헤드 수 ← 원본 4에서 2로 축소
head_dim = n_embd // n_head  # 각 헤드의 차원 = 4 // 2 = 2

print(f"  n_layer    = {n_layer}")
print(f"  n_embd     = {n_embd}")
print(f"  block_size = {block_size}")
print(f"  n_head     = {n_head}")
print(f"  head_dim   = {head_dim} (= n_embd // n_head = {n_embd} // {n_head})")

# --- 원본 코드 (microgpt.py L79~88) ---
# matrix: nout×nin 크기의 2D 리스트를 만드는 람다 함수
# 각 원소는 Value(가우시안 랜덤), std=0.08로 작은 값에서 시작
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte': matrix(vocab_size, n_embd),      # 토큰 임베딩 (27 × 4)
    'wpe': matrix(block_size, n_embd),       # 위치 임베딩 (8 × 4)
    'lm_head': matrix(vocab_size, n_embd),   # 출력 프로젝션 (27 × 4)
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)    # 쿼리 가중치 (4 × 4)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)    # 키 가중치   (4 × 4)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)    # 값 가중치   (4 × 4)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)    # 출력 가중치 (4 × 4)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd) # MLP 업 (16 × 4)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd) # MLP 다운 (4 × 16)

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"\n  총 파라미터 수: {len(params)}")

# ─── 디버그 포인트: state_dict 구조와 각 행렬의 shape ───────────────────────
print(f"\n  [state_dict 구조 — 행렬 이름과 shape]")
for name, mat in state_dict.items():
    rows = len(mat)
    cols = len(mat[0]) if mat else 0
    num_params = rows * cols
    # 첫 번째 원소의 실제 값도 보여줌
    sample_val = mat[0][0].data if mat and mat[0] else None
    print(f"    {name:25s} → shape ({rows:2d} × {cols:2d}) = {num_params:4d} params  (예: {sample_val:+.4f})")

# ─── 핵심 관찰: 임베딩 테이블 ──────────────────────────────────────────────
print(f"\n  [토큰 임베딩 관찰 — wte에서 BOS의 벡터]")
bos_emb = state_dict['wte'][BOS]
print(f"    wte[BOS={BOS}] = [{', '.join(f'{v.data:+.4f}' for v in bos_emb)}]")
print(f"    → 이 {n_embd}차원 벡터가 BOS 토큰의 '신경 서명(neural signature)'")

a_emb = state_dict['wte'][0]  # 'a' = 0
print(f"    wte['a'=0]  = [{', '.join(f'{v.data:+.4f}' for v in a_emb)}]")
print(f"    → 처음엔 랜덤이지만, 학습하면서 의미 있는 표현으로 변한다")


# =============================================================================
# 섹션 2: 모델 아키텍처 — 헬퍼 함수들 (microgpt.py L93~105)
# =============================================================================
#
# [explain.md - Architecture 섹션 요약]
# 모델 아키텍처는 상태 없는(stateless) 함수:
# 토큰, 위치, 파라미터, 이전 위치의 캐시된 key/value를 받아서
# 다음에 올 토큰에 대한 로짓(점수)을 반환한다.
#
# GPT-2를 따르되 약간의 단순화:
# - LayerNorm → RMSNorm
# - bias 없음
# - GeLU → ReLU
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 60)
print("📐 [섹션 2] 헬퍼 함수: linear, softmax, rmsnorm")
print("=" * 60)

# --- 원본 코드 (microgpt.py L93~94) ---
# [explain.md 설명]
# linear은 행렬-벡터 곱셈이다.
# 벡터 x와 가중치 행렬 w를 받아, w의 각 행과 x의 내적을 계산한다.
# 이것이 신경망의 기본 빌딩 블록: 학습된 선형 변환.
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

# --- 원본 코드 (microgpt.py L96~100) ---
# [explain.md 설명]
# softmax는 로짓(raw score) 벡터를 확률 분포로 변환:
# - 모든 값이 [0, 1] 사이로 가고, 합이 1이 된다.
# - 최댓값을 먼저 빼는 이유: 수치 안정성 (exp의 오버플로우 방지)
# - 수학적으로는 결과가 동일하다.
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

# --- 원본 코드 (microgpt.py L102~105) ---
# [explain.md 설명]
# rmsnorm (Root Mean Square Normalization):
# 벡터를 단위 RMS를 갖도록 재조정(rescale)한다.
# 활성값이 네트워크를 통과하면서 커지거나 줄어드는 것을 방지 → 학습 안정화.
# 원본 GPT-2의 LayerNorm의 더 단순한 변형이다.
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

# ─── 디버그 포인트: 각 헬퍼 함수의 입출력 관찰 ─────────────────────────────

# linear 테스트
print(f"\n  [linear 함수 관찰]")
test_x = [Value(1.0), Value(2.0), Value(3.0)]
test_w = [[Value(0.1), Value(0.2), Value(0.3)],
           [Value(0.4), Value(0.5), Value(0.6)]]
test_out = linear(test_x, test_w)
print(f"    입력 x:  [{', '.join(f'{v.data:.1f}' for v in test_x)}]  (길이 {len(test_x)})")
print(f"    가중치 w: {len(test_w)}×{len(test_w[0])} 행렬")
print(f"    출력:    [{', '.join(f'{v.data:.2f}' for v in test_out)}]  (길이 {len(test_out)})")
print(f"    검증: w[0]·x = 0.1×1 + 0.2×2 + 0.3×3 = {0.1*1 + 0.2*2 + 0.3*3:.1f} ✓")
print(f"    💡 출력 길이 = w의 행 수 (nout). 입력을 다른 차원으로 '투영(project)'하는 것!")

# softmax 테스트
print(f"\n  [softmax 함수 관찰]")
test_logits = [Value(2.0), Value(5.0), Value(1.0)]
test_probs = softmax(test_logits)
print(f"    입력 logits: [{', '.join(f'{v.data:.1f}' for v in test_logits)}]")
print(f"    출력 probs:  [{', '.join(f'{v.data:.4f}' for v in test_probs)}]")
print(f"    합계: {sum(p.data for p in test_probs):.6f} (= 1.0이어야 함)")
print(f"    💡 가장 큰 logit(5.0) → 가장 높은 확률({max(p.data for p in test_probs):.4f})")

# rmsnorm 테스트
print(f"\n  [rmsnorm 함수 관찰]")
test_vec = [Value(3.0), Value(4.0), Value(0.0), Value(1.0)]
normed = rmsnorm(test_vec)
print(f"    입력:  [{', '.join(f'{v.data:.1f}' for v in test_vec)}]")
print(f"    출력:  [{', '.join(f'{v.data:.4f}' for v in normed)}]")
rms_before = (sum(v.data**2 for v in test_vec) / len(test_vec))**0.5
rms_after = (sum(v.data**2 for v in normed) / len(normed))**0.5
print(f"    RMS 변환: {rms_before:.4f} → {rms_after:.4f} (≈ 1.0이어야 함)")
print(f"    💡 값의 크기를 정규화해서 학습이 안정되게 한다")


# =============================================================================
# 섹션 3: GPT 모델 함수 (microgpt.py L107~143)
# =============================================================================
#
# [explain.md - Architecture 섹션 요약]
# gpt() 함수는 하나의 토큰(token_id)을 특정 시간 위치(pos_id)에서 처리하고,
# 이전 반복의 활성값(keys, values = KV Cache)을 사용한다.
#
# 처리 과정:
# 1. Embeddings (임베딩):
#    - 원시 토큰 ID를 직접 처리할 수 없으므로 벡터(숫자 리스트)로 변환
#    - 토큰 임베딩(wte) + 위치 임베딩(wpe) = 토큰이 무엇이고 어디 있는지 표현
#
# 2. Attention block (어텐션 블록):
#    - Query(Q): "내가 찾는 것은?"
#    - Key(K): "내가 가진 것은?"
#    - Value(V): "선택되면 내가 제공하는 것은?"
#    - 어텐션은 토큰 t가 과거 위치 0..t-1을 "보는" 유일한 장소
#    - 어텐션 = 토큰 간 통신 메커니즘
#
# 3. MLP block (MLP 블록):
#    - 2층 피드포워드 네트워크: 4배로 확장 → ReLU → 원래로 축소
#    - 위치별 독립적인 "사고" 수행
#    - 트랜스포머 = 통신(Attention) + 계산(MLP) 교차 배치
#
# 4. Residual connections (잔차 연결):
#    - 어텐션/MLP 출력을 자신의 입력에 더함 (x = a + b)
#    - 그래디언트가 네트워크를 직접 흐를 수 있게 → 깊은 모델 학습 가능
#
# 5. Output (출력):
#    - 최종 숨겨진 상태를 어휘 크기로 투영 (lm_head)
#    - 27개 숫자(로짓) 출력. 높을수록 모델이 해당 토큰이 다음에 올 것이라 생각
# ─────────────────────────────────────────────────────────────────────────────

# --- 원본 코드 (microgpt.py L107~143) ---
def gpt(token_id, pos_id, keys, values):
    # ─── 1. 임베딩 ──────────────────────────────────────────────
    tok_emb = state_dict['wte'][token_id]  # 토큰 임베딩 룩업
    pos_emb = state_dict['wpe'][pos_id]    # 위치 임베딩 룩업
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # 토큰 + 위치 임베딩 합산
    x = rmsnorm(x)  # 정규화 (잔차 연결 때문에 필요)

    for li in range(n_layer):
        # ─── 2. Multi-head Attention 블록 ────────────────────────
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])  # 쿼리
        k = linear(x, state_dict[f'layer{li}.attn_wk'])  # 키
        v = linear(x, state_dict[f'layer{li}.attn_wv'])  # 값
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]  # 잔차 연결

        # ─── 3. MLP 블록 ────────────────────────────────────────
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])  # 4배 확장
        x = [xi.relu() for xi in x]                       # 비선형 활성화
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])  # 원래 차원으로 축소
        x = [a + b for a, b in zip(x, x_residual)]  # 잔차 연결

    # ─── 4. 출력 프로젝션 ────────────────────────────────────────
    logits = linear(x, state_dict['lm_head'])
    return logits


# =============================================================================
# 섹션 4: "emma" 첫 토큰(BOS)을 GPT에 넣어보기 — 한 줄씩 추적
# =============================================================================

print("\n\n" + "=" * 60)
print("🔬 [섹션 3] 'emma' 처리 — BOS 토큰부터 한 줄씩 추적")
print("=" * 60)

word = "emma"
tokens = [BOS] + [uchars.index(ch) for ch in word] + [BOS]
print(f"\n  '{word}' 토큰열: {tokens}")
print(f"  → [BOS, 'e', 'm', 'm', 'a', BOS] = [{', '.join(str(t) for t in tokens)}]")

# KV 캐시 초기화
keys = [[] for _ in range(n_layer)]
values_cache = [[] for _ in range(n_layer)]

# ─── 첫 번째 토큰 (BOS, pos=0) 처리 과정을 한 줄씩 ────────────────────────
token_id, pos_id = tokens[0], 0
print(f"\n  === 토큰 0 처리: token_id={token_id} (BOS), pos_id={pos_id} ===")

# Step 1: 임베딩 룩업
tok_emb = state_dict['wte'][token_id]
pos_emb = state_dict['wpe'][pos_id]
print(f"\n  [Step 1] 임베딩 룩업")
print(f"    tok_emb = wte[{token_id}] = [{', '.join(f'{v.data:+.4f}' for v in tok_emb)}]")
print(f"    pos_emb = wpe[{pos_id}]  = [{', '.join(f'{v.data:+.4f}' for v in pos_emb)}]")

x = [t + p for t, p in zip(tok_emb, pos_emb)]
print(f"    x = tok + pos    = [{', '.join(f'{v.data:+.4f}' for v in x)}]")

x = rmsnorm(x)
print(f"    x = rmsnorm(x)   = [{', '.join(f'{v.data:+.4f}' for v in x)}]")

# Step 2: Attention
print(f"\n  [Step 2] Multi-head Attention (layer 0)")
x_residual = x
x_norm = rmsnorm(x)
print(f"    x_norm = rmsnorm(x) = [{', '.join(f'{v.data:+.4f}' for v in x_norm)}]")

q = linear(x_norm, state_dict['layer0.attn_wq'])
k = linear(x_norm, state_dict['layer0.attn_wk'])
v = linear(x_norm, state_dict['layer0.attn_wv'])
print(f"    Q = linear(x, Wq) = [{', '.join(f'{v.data:+.4f}' for v in q)}]  ('{word}' 현재 토큰이 '찾는 것')")
print(f"    K = linear(x, Wk) = [{', '.join(f'{v.data:+.4f}' for v in k)}]  ('가진 것')")
print(f"    V = linear(x, Wv) = [{', '.join(f'{v.data:+.4f}' for v in v)}]  ('제공하는 것')")

keys[0].append(k)
values_cache[0].append(v)

print(f"\n    [헤드별 어텐션 가중치]")
x_attn = []
for h in range(n_head):
    hs = h * head_dim
    q_h = q[hs:hs+head_dim]
    k_h = [ki[hs:hs+head_dim] for ki in keys[0]]
    v_h = [vi[hs:hs+head_dim] for vi in values_cache[0]]
    attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
    attn_weights = softmax(attn_logits)
    head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
    x_attn.extend(head_out)

    print(f"    헤드 {h}: Q_h=[{', '.join(f'{v.data:+.4f}' for v in q_h)}]")
    print(f"           attn_weights=[{', '.join(f'{w.data:.4f}' for w in attn_weights)}]")
    print(f"           → 현재 pos=0 (첫 토큰)이라 자기 자신만 볼 수 있음 → 가중치=[1.0]")

x_attn_proj = linear(x_attn, state_dict['layer0.attn_wo'])
x = [a + b for a, b in zip(x_attn_proj, x_residual)]
print(f"\n    attn_out + residual = [{', '.join(f'{v.data:+.4f}' for v in x)}]")

# Step 3: MLP
print(f"\n  [Step 3] MLP 블록 (layer 0)")
x_residual = x
x_norm = rmsnorm(x)
x_up = linear(x_norm, state_dict['layer0.mlp_fc1'])
print(f"    mlp_fc1 출력 (4배 확장): 길이 {len(x_up)}, 예: [{', '.join(f'{v.data:+.4f}' for v in x_up[:4])}] ...")
x_relu = [xi.relu() for xi in x_up]
num_active = sum(1 for xi in x_relu if xi.data > 0)
print(f"    ReLU 후: {num_active}/{len(x_relu)} 뉴런 활성 (나머지는 '죽은 뉴런')")
x_down = linear(x_relu, state_dict['layer0.mlp_fc2'])
x = [a + b for a, b in zip(x_down, x_residual)]
print(f"    mlp + residual = [{', '.join(f'{v.data:+.4f}' for v in x)}]")

# Step 4: 출력
print(f"\n  [Step 4] 출력 프로젝션 (lm_head)")
logits = linear(x, state_dict['lm_head'])
print(f"    logits (길이 {len(logits)}): 토큰 0~4 → [{', '.join(f'{v.data:+.4f}' for v in logits[:5])}] ...")

# logits → 확률
probs = softmax(logits)

# 상위 5개 토큰 출력
prob_pairs = [(i, p.data) for i, p in enumerate(probs)]
prob_pairs.sort(key=lambda x: -x[1])
print(f"\n    [모델 예측 — Top 5 확률]")
for rank, (idx, prob) in enumerate(prob_pairs[:5]):
    token_str = "[BOS]" if idx == BOS else f"'{uchars[idx]}'"
    print(f"      #{rank+1}: {token_str:6s} (id={idx:2d}) → {prob:.4f}")
print(f"    → 아직 랜덤 파라미터라서 예측이 균등에 가까움")


# =============================================================================
# 섹션 5: "emma" 전체 시퀀스 처리 — 어텐션 가중치 관찰
# =============================================================================

print("\n\n" + "=" * 60)
print("👁️ [섹션 4] 'emma' 전체 시퀀스 — 어텐션 가중치 관찰")
print("=" * 60)

# 전체 시퀀스를 처리하면서 어텐션 가중치 수집
keys2 = [[] for _ in range(n_layer)]
values2 = [[] for _ in range(n_layer)]
all_attn_weights = []  # 위치별 어텐션 가중치 저장

# gpt 함수를 어텐션 가중치를 기록하는 버전으로 수동 실행
for pos_id in range(len(tokens) - 1):
    token_id = tokens[pos_id]
    token_str = "[BOS]" if token_id == BOS else f"'{uchars[token_id]}'"

    # 임베딩
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    # 어텐션
    x_residual = x
    x = rmsnorm(x)
    q = linear(x, state_dict['layer0.attn_wq'])
    k = linear(x, state_dict['layer0.attn_wk'])
    v = linear(x, state_dict['layer0.attn_wv'])
    keys2[0].append(k)
    values2[0].append(v)

    pos_attn = {}
    x_attn = []
    for h in range(n_head):
        hs = h * head_dim
        q_h = q[hs:hs+head_dim]
        k_h = [ki[hs:hs+head_dim] for ki in keys2[0]]
        v_h = [vi[hs:hs+head_dim] for vi in values2[0]]
        attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
        attn_weights_h = softmax(attn_logits)
        head_out = [sum(attn_weights_h[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
        x_attn.extend(head_out)
        pos_attn[h] = [w.data for w in attn_weights_h]

    all_attn_weights.append((pos_id, token_str, pos_attn))

    # MLP
    x = linear(x_attn, state_dict['layer0.attn_wo'])
    x = [a + b for a, b in zip(x, x_residual)]
    x_residual = x
    x = rmsnorm(x)
    x = linear(x, state_dict['layer0.mlp_fc1'])
    x = [xi.relu() for xi in x]
    x = linear(x, state_dict['layer0.mlp_fc2'])
    x = [a + b for a, b in zip(x, x_residual)]

# 어텐션 가중치 출력
token_labels = ["BOS", "e", "m₁", "m₂", "a"]

print(f"\n  [어텐션 가중치 — 각 위치가 과거 어느 토큰에 주목하는가?]")
print(f"  (랜덤 파라미터라서 의미 있는 패턴은 학습 후에 나타남)\n")

for pos_id, token_str, pos_attn in all_attn_weights:
    print(f"  위치 {pos_id} ({token_str:5s}):")
    for h in range(n_head):
        weights = pos_attn[h]
        # 시각화: 막대그래프
        bars = ""
        for t, w in enumerate(weights):
            bar_len = int(w * 20)
            bars += f"    {token_labels[t]:4s} {'█' * bar_len}{'░' * (20 - bar_len)} {w:.3f}\n"
        print(f"    헤드 {h}:")
        print(bars, end="")


# =============================================================================
# 핵심 정리
# =============================================================================
print("=" * 60)
print("🎯 핵심 정리")
print("=" * 60)
print(f"""
  1. 파라미터: 작은 랜덤 숫자로 초기화된 Value들의 행렬 집합
     - wte ({vocab_size}×{n_embd}): 각 토큰의 벡터 표현
     - wpe ({block_size}×{n_embd}): 각 위치의 벡터 표현
     - attn_w* ({n_embd}×{n_embd}): 어텐션 변환 행렬
     - mlp_fc1/fc2: 피드포워드 네트워크
     - lm_head ({vocab_size}×{n_embd}): 벡터 → 로짓 변환

  2. 모델 흐름:
     토큰ID → 임베딩 → rmsnorm → [어텐션 → 잔차] → [MLP → 잔차] → lm_head → 로짓

  3. 어텐션: 현재 토큰이 과거 토큰들을 "보고" 정보를 모으는 통신 메커니즘
     - pos=0: 자기 자신만 볼 수 있음 (가중치 = [1.0])
     - pos=n: 0~n까지의 모든 과거 토큰을 볼 수 있음

  4. 현재는 랜덤 파라미터라서 예측이 무의미
     → 다음 챕터(학습 루프)에서 파라미터를 최적화하면 의미 있는 예측 시작!

  총 파라미터 수: {len(params)} (원본 microgpt: 4,192)
""")
