import cirq
import numpy as np
from fractions import Fraction

# NAQFTのパラメータ
params = [0.3365999162197113, 0.8254781365394592, 0.6756895780563354,
          0.042534347623586655, 0.14197996258735657, 0.3458109200000763,
          -0.08360755443572998, 0.686887800693512, 0.21553224325180054,
          0.7092717885971069, 0.5580384135246277, 0.7080177068710327]

def naqft_circuit(qubits):
    circuit = cirq.Circuit()
    n = len(qubits)
    
    for i in range(n):
        circuit.append(cirq.rx(params[3*i])(qubits[i]))
        circuit.append(cirq.ry(params[3*i+1])(qubits[i]))
        circuit.append(cirq.rz(params[3*i+2])(qubits[i]))
    
    return circuit

# 関数fの評価を行う量子回路
def modular_exponentiation(a, n):
    def oracle(qr, x):
        if (a**x) % n == 1:
            return 1
        return 0

    return oracle

def qpe_amodn(a, n):
    qpe_circuit = cirq.Circuit()
    n_qubits = n.bit_length()
    qregs = cirq.LineQubit.range(n_qubits * 2)

    # アダマールゲートを前半のレジスタに適用
    for q in qregs[:n_qubits]:
        qpe_circuit.append(cirq.H(q))

    # モジュラエクスポネンシャーションを実行
    oracle = modular_exponentiation(a, n)
    for i in range(n_qubits):
        qpe_circuit.append(cirq.ControlledGate(cirq.MatrixGate(oracle(qregs[:n_qubits], 2**i))).on(qregs[n_qubits + i]))

    # NAQFTを適用
    naqft = naqft_circuit(qregs[:n_qubits])
    qpe_circuit.append(naqft)

    return qpe_circuit, qregs

# 測定結果を因数に変換する関数
def get_factors(measurements, n):
    phase = sum([val * (0.5**(i+1)) for i, val in enumerate(measurements)])
    frac = Fraction(phase).limit_denominator(n)
    r = frac.denominator
    if r % 2 == 1:
        r *= 2
    guess_factor1 = gcd(a**(r//2) - 1, n)
    guess_factor2 = gcd(a**(r//2) + 1, n)
    return guess_factor1, guess_factor2

# 実際のショアのアルゴリズムの実行
N = 15  # 因数分解対象の整数
a = 7   # 使用した基数

qpe_circuit, qregs = qpe_amodn(a, N)
qpe_circuit.append(cirq.measure(*qregs[:N.bit_length()], key='result'))

# 回路の出力
print("Shor's Algorithm with NAQFT Circuit:")
print(qpe_circuit)

# シミュレーション
simulator = cirq.Simulator()
result = simulator.run(qpe_circuit, repetitions=10)
print("Simulation Result:")
print(result)

# 測定結果の解析
counts = result.histogram(key='result')
print("Counts:", counts)

# 最大カウントを持つ結果を選択
max_count_key = max(counts, key=counts.get)
max_count_result = [int(digit) for digit in format(max_count_key, '0' + str(N.bit_length()) + 'b')]
print("Max Count Measurement:", max_count_result)

# 因数の計算
factor1, factor2 = get_factors(max_count_result, N)
print("Factors:", factor1, factor2)
