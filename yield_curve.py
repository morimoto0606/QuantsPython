import numpy as np
from scipy import optimize as op


def piecewise_linear_yield_curve(base_zero_rates):
    """
    線形補間したdf関数を返す
    :param base_zero_rates: baseとなるグリッドとゼロレートの辞書
    :return: df 関数
    """
    def df(tenor):
        y = np.interp(tenor,
                      list(base_zero_rates.keys()),
                      list(base_zero_rates.values()))
        return np.exp(-tenor * y)

    return df

def swap_value_function(solved_zero_rates,
                        target_maturity,
                        target_swap_rate):
    """
    solved_zero ratesを補間し、target maturityの期間でtarget swaprateのクーポンすスワップの時価関数をかえす。
    :param solved_zero_rates: dictionary of (solved grid , value) 
    :param target_maturity:  target swap maturity
    :param target_swap_rate: target swap rate
    :return: ターゲットグリッドのイールドを引数としたスワップの時価関数
    """
    def get_pv(zero_rate_on_target_grid):
        """
        ターゲットグリッドのイールドを引数としたスワップの時価関数
        :param zero_rate_on_target_grid: ターゲットグリッドのイールド（nd array）
        :return: payer'sの価値
        """
        # solved gridに、新しいグリッドと仮置きしたそのグリッドのイールドを追加
        solved_zero_rates.update(
            {target_maturity: zero_rate_on_target_grid[0]})
        # 追加されたゼロレート
        yield_curve = piecewise_linear_yield_curve(solved_zero_rate)
        value = yield_curve(target_maturity)

        # SwapのCash flow計算
        grid = 1
        while tau * grid <= target_maturity:
            value += target_swap_rate * tau * yield_curve(tau * grid)
            grid += 1
        return 1 - value

    return get_pv


if __name__ == '__main__':
    # grid 間隔を設定
    tau = 0.5
    # benchmarkとするスワップれーとを設定（Andersen Piterberg参照）
    maturities = [1, 2, 3, 5, 7, 10, 12, 15, 20, 25]
    swap_rates = np.array([4.2, 4.3, 4.7, 5.4, 5.7, 6, 6.1, 5.9, 5.6, 5.55]) / 100
    benchmark = dict(zip(maturities, swap_rates))

    # 一番最初のグリッド（１Y）だけチェック
    solved_zero_rate = {}
    swap_value = swap_value_function(solved_zero_rate,
                                     1,
                                     benchmark[1])
    rate0 = swap_value([0.02])
    rate = op.fsolve(swap_value, 0.01)
    print(rate)
    print(swap_value(rate))

    # 全グリッドでzero rateのCalibration
    solved_zero_rate = {}
    for maturity in maturities:
        func = swap_value_function(solved_zero_rate,
                                   maturity,
                                   benchmark[maturity])
        zero_rate = op.fsolve(func, 0.01)
        solved_zero_rate.update({maturity: zero_rate[0]})

    print(solved_zero_rate)

    # Calibration結果をもとにカーブを作成
    curve = piecewise_linear_yield_curve(solved_zero_rate)
    for grid in range(0, 10):
        print(curve(grid))
