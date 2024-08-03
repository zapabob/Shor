import pennylane as qml
from pennylane import numpy as np
from fractions import Fraction
import math
from typing import List, Tuple
from tqdm import tqdm
import logging

# ロギングの設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SYKHamiltonian:
    @staticmethod
    def create(n_qubits: int, J: np.ndarray) -> qml.Hamiltonian:
        """SYKモデルのハミルトニアンを生成する"""
        obs = []
        coeffs = []
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                for k in range(j + 1, n_qubits):
                    for l in range(k + 1, n_qubits):
                        obs.append(
                            qml.PauliX(i)
                            @ qml.PauliY(j)
                            @ qml.PauliZ(k)
                            @ qml.PauliX(l)
                        )
                        coeffs.append(J[i, j, k, l])
        return qml.Hamiltonian(coeffs, obs)


class NAQFTSYKLayer:
    @staticmethod
    def apply(qubits: List[int], J: np.ndarray, time: float) -> None:
        """NAQFTレイヤー（SYKモデル使用）を適用する"""
        n_qubits = len(qubits)
        # SWAPゲート（前半）
        for i in range(n_qubits // 2):
            qml.SWAP(wires=[qubits[i], qubits[-i - 1]])

        # SYKハミルトニアン部分
        H = SYKHamiltonian.create(n_qubits, J)
        qml.exp(H * time)

        # SWAPゲート（後半）
        for i in range(n_qubits // 2):
            qml.SWAP(wires=[qubits[i], qubits[-i - 1]])


class QFT:
    @staticmethod
    def inverse(wires: List[int]) -> None:
        """逆量子フーリエ変換を適用する"""
        for i in reversed(range(len(wires))):
            qml.Hadamard(wires[i])
            for j in range(i):
                qml.ControlledPhaseShift(
                    -np.pi / 2 ** (i - j), wires=[wires[j], wires[i]]
                )


class ModularMultiplication:
    @staticmethod
    def create_unitary(
        a: int, power: int, mod: int = 21, n_qubits: int = 5
    ) -> np.ndarray:
        """modulo 21での制御ユニタリ操作を生成する"""
        if a not in [2, 4, 5, 8, 10, 11, 13, 16, 17, 19, 20]:  # 21と互いに素な数
            raise ValueError(f"'a' must be coprime with 21. Got {a}")

        N = 2**n_qubits
        U = np.eye(N)
        for i in range(mod):
            U[i, (a * i) % mod] = 1
            U[i, i] = 0

        return np.linalg.matrix_power(U, power)


class ShorCircuit:
    @staticmethod
    def create_naqft_syk(
        n_counting_qubits: int, a: int, time: float, J: np.ndarray
    ) -> qml.QNode:
        """NAQFT-SYK版ショアのアルゴリズム回路を生成する"""
        n_target_qubits = 5  # 21の因数分解には5量子ビットが必要
        dev = qml.device("default.qubit", wires=n_counting_qubits + n_target_qubits)

        @qml.qnode(dev)
        def circuit():
            # 初期化
            for i in range(n_counting_qubits):
                qml.Hadamard(wires=i)
            qml.PauliX(wires=n_counting_qubits)

            # 位相推定部分
            for i in range(n_counting_qubits):
                qml.ControlledQubitUnitary(
                    ModularMultiplication.create_unitary(
                        a, 2**i, mod=21, n_qubits=n_target_qubits
                    ),
                    control_wires=[i],
                    wires=list(
                        range(n_counting_qubits, n_counting_qubits + n_target_qubits)
                    ),
                )

            # NAQFT-SYKレイヤーを適用
            NAQFTSYKLayer.apply(range(n_counting_qubits), J, time)

            # 逆QFT
            QFT.inverse(range(n_counting_qubits))

            # 測定
            return qml.probs(wires=range(n_counting_qubits))

        return circuit

    @staticmethod
    def create_standard(n_counting_qubits: int, a: int) -> qml.QNode:
        """標準的なショアのアルゴリズム回路を生成する"""
        n_target_qubits = 5  # 21の因数分解には5量子ビットが必要
        dev = qml.device("default.qubit", wires=n_counting_qubits + n_target_qubits)

        @qml.qnode(dev)
        def circuit():
            # 初期化
            for i in range(n_counting_qubits):
                qml.Hadamard(wires=i)
            qml.PauliX(wires=n_counting_qubits)

            # 位相推定部分
            for i in range(n_counting_qubits):
                qml.ControlledQubitUnitary(
                    ModularMultiplication.create_unitary(
                        a, 2**i, mod=21, n_qubits=n_target_qubits
                    ),
                    control_wires=[i],
                    wires=list(
                        range(n_counting_qubits, n_counting_qubits + n_target_qubits)
                    ),
                )

            # 逆QFT
            QFT.inverse(range(n_counting_qubits))

            # 測定
            return qml.probs(wires=range(n_counting_qubits))

        return circuit


class ResultProcessor:
    @staticmethod
    def process(
        results: np.ndarray, n_counting_qubits: int, a: int, N: int = 21
    ) -> List[int]:
        """結果を処理し、位相と因数を推定する"""
        max_prob_index = np.argmax(results)
        phase = max_prob_index / (2**n_counting_qubits)

        # 連分数展開で有理数近似
        frac = Fraction(phase).limit_denominator(N)
        r = frac.denominator

        if r % 2 == 0:
            guesses = [math.gcd(a ** (r // 2) - 1, N), math.gcd(a ** (r // 2) + 1, N)]
            return [guess for guess in guesses if guess not in [1, N]]
        else:
            return []


class AlgorithmComparator:
    @staticmethod
    def compare(
        n_counting_qubits: int, a: int, time: float, J: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """NAQFT-SYK版と標準版のショアのアルゴリズムを比較する"""
        naqft_circuit = ShorCircuit.create_naqft_syk(n_counting_qubits, a, time, J)
        standard_circuit = ShorCircuit.create_standard(n_counting_qubits, a)

        naqft_results = naqft_circuit()
        standard_results = standard_circuit()

        return naqft_results, standard_results

    @staticmethod
    def run_comparison(
        n_counting_qubits: int, a: int, time: float, J: np.ndarray, n_trials: int = 100
    ) -> Tuple[float, float]:
        """複数回の試行で両アルゴリズムの成功率を比較する"""
        naqft_success = 0
        standard_success = 0

        for _ in tqdm(range(n_trials), desc="Comparing algorithms"):
            naqft_results, standard_results = AlgorithmComparator.compare(
                n_counting_qubits, a, time, J
            )

            naqft_factors = ResultProcessor.process(
                naqft_results, n_counting_qubits, a, N=21
            )
            if naqft_factors:
                naqft_success += 1

            standard_factors = ResultProcessor.process(
                standard_results, n_counting_qubits, a, N=21
            )
            if standard_factors:
                standard_success += 1

        return naqft_success / n_trials, standard_success / n_trials


class ResultPrinter:
    @staticmethod
    def print_comparison_results(
        naqft_results: np.ndarray,
        standard_results: np.ndarray,
        naqft_success_rate: float,
        standard_success_rate: float,
    ) -> None:
        """比較結果を出力する"""
        logging.info("NAQFT-SYK版ショアのアルゴリズム測定結果の確率分布:")
        logging.info(naqft_results)
        logging.info("\n標準的なショアのアルゴリズムの測定結果の確率分布:")
        logging.info(standard_results)

        max_prob_naqft = np.max(naqft_results)
        max_prob_standard = np.max(standard_results)

        logging.info(f"\nNAQFT-SYK版の最大確率: {max_prob_naqft:.4f}")
        logging.info(f"標準版の最大確率: {max_prob_standard:.4f}")

        if max_prob_naqft > max_prob_standard:
            improvement = (max_prob_naqft - max_prob_standard) / max_prob_standard * 100
            logging.info(
                f"NAQFT-SYK版は標準版より{improvement:.2f}%高い最大確率を示しました。"
            )
        elif max_prob_naqft < max_prob_standard:
            deterioration = (
                (max_prob_standard - max_prob_naqft) / max_prob_standard * 100
            )
            logging.info(
                f"NAQFT-SYK版は標準版より{deterioration:.2f}%低い最大確率を示しました。"
            )
        else:
            logging.info("両版の最大確率は同じでした。")

        entropy_naqft = -np.sum(naqft_results * np.log2(naqft_results + 1e-10))
        entropy_standard = -np.sum(standard_results * np.log2(standard_results + 1e-10))

        logging.info(f"\nNAQFT-SYK版のエントロピー: {entropy_naqft:.4f}")
        logging.info(f"標準版のエントロピー: {entropy_standard:.4f}")

        if entropy_naqft < entropy_standard:
            logging.info(
                "NAQFT-SYK版はより低いエントロピーを示し、より集中した確率分布を持っています。"
            )
        elif entropy_naqft > entropy_standard:
            logging.info(
                "NAQFT-SYK版はより高いエントロピーを示し、より分散した確率分布を持っています。"
            )
        else:
            logging.info("両版のエントロピーは同じです。")

        logging.info(f"\n複数回の試行における成功率:")
        logging.info(f"NAQFT-SYK版の成功率: {naqft_success_rate:.2%}")
        logging.info(f"標準版の成功率: {standard_success_rate:.2%}")

        if standard_success_rate == 0:
            if naqft_success_rate > 0:
                logging.info(
                    "NAQFT-SYK版は標準版よりも優れた性能を示しました。標準版は成功しませんでした。"
                )
            else:
                logging.info("両方のバージョンとも成功しませんでした。")
        elif naqft_success_rate > standard_success_rate:
            improvement = (
                (naqft_success_rate - standard_success_rate)
                / standard_success_rate
                * 100
            )
            logging.info(
                f"NAQFT-SYK版は標準版より{improvement:.2f}%高い成功率を示しました。"
            )
        elif naqft_success_rate < standard_success_rate:
            deterioration = (
                (standard_success_rate - naqft_success_rate)
                / standard_success_rate
                * 100
            )
            logging.info(
                f"NAQFT-SYK版は標準版より{deterioration:.2f}%低い成功率を示しました。"
            )
        else:
            logging.info("両版の成功率は同じでした。")


def main():
    n_counting_qubits = 8
    a = 2  # 21と互いに素な数
    time = 1.0
    J = np.random.normal(
        0,
        np.sqrt(2 / n_counting_qubits**3),
        (n_counting_qubits, n_counting_qubits, n_counting_qubits, n_counting_qubits),
    )
    n_trials = 100

    logging.info(
        "NAQFT-SYK版ショアのアルゴリズムと標準版の比較を開始します（21の因数分解）..."
    )
    naqft_results, standard_results = AlgorithmComparator.compare(
        n_counting_qubits, a, time, J
    )
    naqft_success_rate, standard_success_rate = AlgorithmComparator.run_comparison(
        n_counting_qubits, a, time, J, n_trials
    )

    ResultPrinter.print_comparison_results(
        naqft_results, standard_results, naqft_success_rate, standard_success_rate
    )


if __name__ == "__main__":
    main()
