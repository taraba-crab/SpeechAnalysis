# 音声分析の勉強

## 目的

Pythonによる音声分析の勉強。

## 目標

Jypter Notebook、各種ライブラリを使って音声データを加工し、保存する。

音声の特徴とかが出るようなデータになれば最高。

## フォルダ構成

requirements.txtは根に置いて随時必要なライブラリを更新していく。(インストールするときは`pip install -r requirements.txt`)

時系列順に勉強内容を振り返りたいので一区切りするごとに日付でフォルダを切っていく。

### 日付フォルダ

日付フォルダ内のREADME.mdに目標を書く。

日付フォルダ内では取り組み内容ごとにフォルダを切っていく。

以下の観点で日付フォルダの目標は修正しても良い。

* 達成するのが能力的に厳しい
* 達成までに時間がかかりそう(2, 3週間必要とか)
* やっていることは大事だが、目標からやや逸れている

## ブランチとマージ

日付フォルダでの取り組みごとにブランチを切り、何か大きめなことをした段階でプルリクエストを行い、スカッシュでマージする。

マージしたブランチは削除する。

## 参考書籍

* 概念を大切にする微積分―1変数
* 行列プログラマー: Pythonプログラムで学ぶ線形代数
* 解析学概論
* 音声認識 (機械学習プロフェッショナルシリーズ) 
* Pythonではじめる教師なし学習――機械学習の可能性を広げるラベルなしデータの利用

---

* docs
    * 2020-11-12
        * [free.html](.\2020-11-12\free.html)
        * coordinate_and_basis
            * [座標と基底.html](.\2020-11-12\coordinate_and_basis\座標と基底.html)
        * fft
            * [高速フーリエ変換.html](.\2020-11-12\fft\高速フーリエ変換.html)
        * sin
            * [sin直交性.html](.\2020-11-12\sin\sin直交性.html)
        * stopwatch
            * [正弦波の重み付き和としての信号.html](.\2020-11-12\stopwatch\正弦波の重み付き和としての信号.html)
    * 2020-11-26
        * high_frequency_emphasis
            * [高域強調.html](.\2020-11-26\high_frequency_emphasis\高域強調.html)
        * math_functions
            * [関数の感覚.html](.\2020-11-26\math_functions\関数の感覚.html)
        * sound
            * [メル尺度.html](.\2020-11-26\sound\メル尺度.html)
            * [音圧レベル.html](.\2020-11-26\sound\音圧レベル.html)
        * window_function
            * [窓関数.html](.\2020-11-26\window_function\窓関数.html)
    * 2020-12-10
        * stft
            * [短時間フーリエ変換のためのコード.html](.\2020-12-10\stft\短時間フーリエ変換のためのコード.html)
            * [音声の短時間フーリエ変換.html](.\2020-12-10\stft\音声の短時間フーリエ変換.html)
            * [高域強調から短時間フーリエ変換.html](.\2020-12-10\stft\高域強調から短時間フーリエ変換.html)
    * 2020-12-17
        * cepstrum
            * [ケプストラム特徴量.html](.\2020-12-17\cepstrum\ケプストラム特徴量.html)
        * melfilterbank
            * [メルフィルタバンク.html](.\2020-12-17\melfilterbank\メルフィルタバンク.html)

---

basis.py
```py
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

def _linearly_independent(*vecs):
    m = np.stack(vecs)
    return LA.det(m) != 0

class Basis:
    def __init__(self, *vecs):
        assert _linearly_independent(vecs), "Basis requires that vectors are each linearly independent."
        self.basis_mat = np.stack(vecs, axis=1)
        self.dimension = len(vecs[0])

    def coordinate(self, vec):
        return self.basis_mat.dot(vec)
    
    def plot_coordinate(self, p, n=10):
        assert self.dimension == 2, "method 'plot_coordinate' is implemented for only 2-dimension."

        plt.xlim(0, n)
        plt.ylim(0, n)

        for x in range(-n, n):
            start = self.coordinate([x, -n])
            end = self.coordinate([x, n])

            xs = np.linspace(start[0], end[0])
            ys = np.linspace(start[1], end[1])
            plt.plot(xs, ys, color="#AAAAAA")
            
        for y in range(-n, n):
            start = self.coordinate([-n, y])
            end = self.coordinate([n, y])

            xs = np.linspace(start[0], end[0])
            ys = np.linspace(start[1], end[1])
            plt.plot(xs, ys, color="#AAAAAA")

        plt.plot(*self.coordinate(p), marker="o")

```

fft.py
```py
import numpy as np

def fft(w, s):
    n = len(s)
    if n == 1: return s

    f0 = fft(w*w, s[::2])
    f1 = fft(w*w, s[1::2])

    return np.concatenate([[f0[j] + w**j*f1[j] for j in range(n//2)],
                           [f0[j] + w**(n/2+j)*f1[j] for j in range(n//2)]])
```

highpathfilter.py
```py
import numpy as np

def high_pass_filter(a, fs):
    def h(f):
        w = 2 * np.pi * f / fs
        z = np.exp(-1j * w)
        return 1 - a * z

    return h

```

mel.py
```py
import numpy as np

def mel_scale(f0):
    m0 = 1000.0 / np.log10(1000.0 / f0 + 1.0)

    def mel_scaled(f):
        return m0 * np.log10(f / f0 + 1.0)

    return mel_scaled
```

SPL.py
```py
import numpy as np

def spl(p):
    p0 = 20 * (10**6)
    return 20 * np.log10(p/p0)

```

window_function.py
```py
import numpy as np
import matplotlib.pyplot as plt

# 2秒を2000個に離散化
SEC = 2
N = 2000
S = np.linspace(0, SEC, N)

def ms_to_n(ms):
    """
    ミリ秒をnに変換
    """

    unit = N // SEC // 1000
    return ms // unit

def sin_wave(k):
    """
    k(Hz)のsin波
    """

    w = 2 * np.pi * k
    f = np.sin(S * w)
    return f


def plot_sample_wave(hz, ms):
    """
    周波数hzのsin波のmsミリ秒までを表現するグラフをプロットする
    """

    s = sin_wave(hz)
    plt.plot(S, s)

    th = ms_to_n(ms)
    plt.plot(S[:th], s[:th])


def rectangle_window(n):
    """
    全体をNとして、nまで切り取る矩形窓
    """

    window = np.zeros(N)
    window[:n] = 1
    return window


def hamming_window(n):
    """
    全体をNとして、nまで切り取るハミング窓
    """

    f = np.zeros(N)
    window = np.repeat(0.54, n) - 0.46 * np.cos(2 * np.pi / n * np.arange(n))
    f[:n] = window
    return f


def plot_window(n, window_func):
    """
    窓関数をプロットする。わかりやすくするためにマイナスの余分な領域もプロット
    """

    extra = 100
    xs = np.arange(-extra, N)
    rect = np.concatenate([np.zeros(extra), window_func(n)])
    plt.plot(xs, rect)


def plot_clipped(hz, ms, window_func):
    """
    sin波のmsまでのところまでを切り取る
    """
    
    n = ms_to_n(ms)
    s = sin_wave(hz)
    window = window_func(n)
    clipped = s * window
    plt.plot(S, clipped)

```

audio.py
```py
import numpy as np
import matplotlib.pyplot as plt
import itertools

def time_axis(ary, rate: np.float64) -> np.ndarray:
    """NumPy配列とサンプリング周波数から時間軸を取得する。

    Args:
        ary (numpy.ndarray[int16]): 任意の配列
        rate (numpy.float64): サンプリング周波数

    Returns:
        numpy.ndarray[numpy.float64]: aryと同じ要素数を持つ時間軸
    """
    
    return np.arange(len(ary)) / rate


class Audio:
    """音声分析用クラス
    
    Attributes:
        rate (int): サンプリング周波数
        data (numpy.ndarray[numpy.int16]): 1次元のデータ
        times (numpy.ndarray[numpy.int16]): 時間軸
    """
    
    def __init__(self, rate: int, data: np.ndarray):
        self.data = data
        self.rate = rate
        self.times = time_axis(self.data, self.rate)

    def plot(self):
        """横軸を時間軸、縦軸をオーディオデータとしたグラフをプロットする。"""
        
        plt.plot(self.times, self.data)
        
    def each_frame(self, n_frame: int, step_ms: int):
        """オーディオデータをn_frameのフレーム長で約step_ms(ミリ秒)ずつずらしながら切り取っていくジェネレータを取得する。
        
        Args:
            n_frame (int): フレーム長
            step_ms (int): ずらす長さ(ミリ秒)

        Yields:
            numpy.ndarray[numpy.int16]: オーディオデータをフレーム長で切り取ったNumPy配列。
            step_ms(ミリ秒)ずつずらしながら切り取っていく。
            余った部分は取得しない。
        """
        
        step = self.rate * step_ms // 1000
        
        i = 0
        n = len(self.data)
        while i+n_frame <= n:
            yield self.data[i:i+n_frame]
            i += step

    def high_pass_filtered(self):
        return Audio(self.rate, high_pass_filter(self.data))


def high_pass_filter(data, a=0.97):
    n = len(data)
    y = [None] * n
    y[0] = data[0]
    for i in range(1, n):
        y[i] = data[i] - a * data[i-1]

    return np.array(y)


def frame_candidates(rate: int, min_ms: int, max_ms: int):
    """サンプリング周波数から、窓関数で切り取るフレーム長の候補のジェネレータを取得する。

    Args:
        rate (int): サンプリング周波数

    Yields:
        int: 候補となるミリ秒
    """
    min_n = rate * min_ms / 1000
    max_n = rate * max_ms / 1000

    for p in itertools.count(1):
        n = 1 << p
        if min_n < n < max_n:
            yield n
        elif n > max_n:
            break


def sin_wave(k: int, rate: int, ms: int):
    """サンプリング周波数rate(Hz)におけるms(ミリ秒)までの周期kのsin波を取得する。

    Args:
        k (int): 周波数
        rate (int): サンプリング周波数
        ms (int): ミリ秒
    
    Returns:
        numpy.ndarray[numpy.float64]: sin波
    """
    
    xs = np.linspace(0, ms / 1000, rate * ms // 1000)
    w = 2 * np.pi * k
    return np.sin(xs * w)


def stft(a: Audio, window, step_length: int):
    """短時間フーリエ変換
    
    step_length(ms)ごとに、オーディオデータ(a)をフレーム長(frame_length)の範囲で切り取っていき、
    それぞれに窓関数(window)を適用し、高速フーリエ変換する。

    Args:
        a (Audio): オーディオ
        window (numpy.ndarray[numpy.float64]): 窓関数
        step_length (int): ずらす長さ(ミリ秒)
    """
    
    for frame in a.each_frame(len(window), step_length):
        windowed = frame * window
        ffted = np.fft.rfft(windowed)
        
        yield ffted

```
