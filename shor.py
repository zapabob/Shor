import tensorflow as tf
import tensornetwork as tn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import time
from fractions import Fraction
from math import gcd

# GPUメモリ使用を制限
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print(f"Using TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.test.is_gpu_available()}")

tn.set_default_backend("tensorflow")

def matrix_power(matrix: tf.Tensor, power: int) -> tf.Tensor:
    result = tf.eye(matrix.shape[0], dtype=tf.complex128)
    for _ in range(power):
        result = tf.matmul(result, matrix)
    return result

def create_qft_tensor(n: int) -> tn.Node:
    i, j = tf.meshgrid(tf.range(2**n, dtype=tf.float64), tf.range(2**n, dtype=tf.float64))
    angle = 2 * np.pi * i * j / 2**n
    tensor = tf.complex(tf.cos(angle), tf.sin(angle)) / tf.sqrt(tf.cast(2**n, tf.float64))
    return tn.Node(tensor)

def create_naqft_tensor(n: int) -> tn.Node:
    i, j = tf.meshgrid(tf.range(2**n, dtype=tf.float64), tf.range(2**n, dtype=tf.float64))
    phase = tf.reduce_sum([((tf.bitwise.right_shift(tf.cast(i, tf.int32), k) & 1) * 
                            (tf.bitwise.right_shift(tf.cast(j, tf.int32), k) & 1) * 
                            2 * np.pi / 2**(k+1)) 
                           for k in range(n)], axis=0)
    tensor = tf.complex(tf.cos(phase), tf.sin(phase)) / tf.sqrt(tf.cast(2**n, tf.float64))
    return tn.Node(tensor)

def quantum_phase_estimation(U: tn.Node, eigenvector: tn.Node, n_qubits: int, use_naqft: bool = False) -> tn.Node:
    # 初期状態の準備
    initial_state = tn.Node(tf.concat([tf.constant([1.0 + 0.j], dtype=tf.complex128), tf.zeros(2**n_qubits - 1, dtype=tf.complex128)], axis=0))
    state = tn.outer_product(initial_state, eigenvector)
    
    # 制御Uゲートの適用
    for i in range(n_qubits):
        U_power = tn.Node(matrix_power(U.tensor, 2**i))
        control_U = tn.Node(tf.eye(2**(n_qubits + eigenvector.tensor.shape[0]), dtype=tf.complex128))
        control_U.tensor = tf.tensor_scatter_nd_update(
            control_U.tensor, 
            [[j, j] for j in range(2**n_qubits, 2**(n_qubits + eigenvector.tensor.shape[0]))],
            tf.reshape(U_power.tensor, [-1])
        )
        state = tn.contract_between(control_U, state)

    # QFTまたはNAQFTの適用
    if use_naqft:
        qft = create_naqft_tensor(n_qubits)
    else:
        qft = create_qft_tensor(n_qubits)
    
    qft_full = tn.Node(tf.eye(2**(n_qubits + eigenvector.tensor.shape[0]), dtype=tf.complex128))
    qft_full.tensor = tf.tensor_scatter_nd_update(
        qft_full.tensor, 
        [[i, j] for i in range(2**n_qubits) for j in range(2**n_qubits)],
        tf.reshape(qft.tensor, [-1])
    )
    state = tn.contract_between(qft_full, state)

    return tn.Node(state.tensor[:2**n_qubits])

def measure_state(state: tn.Node) -> int:
    probabilities = tf.abs(state.tensor)**2
    return tf.random.categorical(tf.math.log([probabilities]), 1)[0, 0].numpy()

def continued_fraction(x: float, max_denominator: int) -> Fraction:
    return Fraction(x).limit_denominator(max_denominator)

def shor_algorithm(N: int, a: int, n_qubits: int, use_naqft: bool = False) -> Tuple[int, float]:
    # Uゲートの定義
    U_tensor = tf.scatter_nd([[i, (a * i) % N] for i in range(N)], 
                             tf.ones(N, dtype=tf.complex128),
                             [N, N])
    U = tn.Node(U_tensor)

    # 固有ベクトルの準備
    eigenvector = tn.Node(tf.ones(N, dtype=tf.complex128) / tf.cast(tf.sqrt(tf.cast(N, tf.float64)), tf.complex128))

    # 量子位相推定
    start_time = time.time()
    final_state = quantum_phase_estimation(U, eigenvector, n_qubits, use_naqft)
    measured_value = measure_state(final_state)
    phase = measured_value / 2**n_qubits

    # 連分数展開
    fraction = continued_fraction(phase, N)
    r = fraction.denominator
    end_time = time.time()

    # rが偶数でない場合、もしくはa^(r/2) ≡ -1 (mod N)の場合、失敗とする
    if r % 2 != 0 or pow(a, r//2, N) == N - 1:
        return 0, end_time - start_time

    # 因数を計算
    factor = gcd(pow(a, r//2, N) - 1, N)
    return factor, end_time - start_time

def run_shor_experiment(N: int, a: int, n_qubits: int, use_naqft: bool = False) -> Tuple[int, float]:
    factor, time_taken = shor_algorithm(N, a, n_qubits, use_naqft)
    return factor, time_taken

def main():
    N = 15  # 素因数分解する数
    a = 7   # 互いに素な数
    n_qubits_range = range(4, 8)  # 4から7量子ビットまでテスト
    
    standard_results = []
    naqft_results = []

    for n in n_qubits_range:
        print(f"Testing with {n} qubits:")
        
        standard_factor, standard_time = run_shor_experiment(N, a, n)
        naqft_factor, naqft_time = run_shor_experiment(N, a, n, use_naqft=True)
        
        standard_results.append((n, standard_factor, standard_time))
        naqft_results.append((n, naqft_factor, naqft_time))
        
        print(f"Standard - Factor: {standard_factor}, Time: {standard_time:.4f} seconds")
        print(f"NAQFT    - Factor: {naqft_factor}, Time: {naqft_time:.4f} seconds")
        print()

    # 結果のプロット
    plot_results(standard_results, naqft_results)

def plot_results(standard_results, naqft_results):
    n_qubits = [r[0] for r in standard_results]
    
    std_factors = [r[1] for r in standard_results]
    naqft_factors = [r[1] for r in naqft_results]
    
    std_times = [r[2] for r in standard_results]
    naqft_times = [r[2] for r in naqft_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(n_qubits, std_factors, 'bo-', label='Standard')
    ax1.plot(n_qubits, naqft_factors, 'ro-', label='NAQFT')
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Found Factor')
    ax1.set_title('Found Factor vs Number of Qubits')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(n_qubits, std_times, 'bo-', label='Standard')
    ax2.plot(n_qubits, naqft_times, 'ro-', label='NAQFT')
    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title('Execution Time vs Number of Qubits')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('shor_results.png')
    plt.close()

    print("\n<ANTARTIFACTLINK identifier='shor-results' type='image/png' title='Shor Algorithm Results' />")

if __name__ == "__main__":
    main()