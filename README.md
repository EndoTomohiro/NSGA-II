# NSGA-II

卒論論文のプログラム

## 卒業論文の概要

NSGA-IIを用いてグリーン配送計画問題を解く

- 配送計画問題：複数の車両を用いて顧客へ荷物を配送する経路のうち、総移動コストが最小になる経路を求める問題
- グリーン配送計画問題：環境に関する要素を配送計画問題に付加した問題
- 遺伝的アルゴリズム(GA)：自然界における生物の遺伝や進化の過程に従って設計・実装されたアルゴリズム
- NSGA-II：GAを改良したアルゴリズム

## パッケージのインストール

`pip install -r requirements.txt`

## 各プログラムおよびfigureフォルダに出力される図の説明

- Costa.py : de Oliveira da Costa らの手法で近似解を出力
  - route_Costa.png : ルート
  - CO2_Costa.png : 辺ごとのCO2排出量

- Jemai.py : Jemai らの手法で近似解を出力

- Proposed.py : 提案手法で近似解を出力
  - route_Proposed.png : ルート
  - CO2_Proposed.png : 辺ごとのCO2排出量

- Proposed_crossover.py : 交叉の前後におけるルートおよび辺ごとのCO2排出量を出力
  - crossover_route_1.png : 交叉前のルート
  - crossover_route_2.png : 交叉後のルート
  - crossover_CO2_1.png : 交叉前の辺ごとのCO2排出量
  - crossover_CO2_2.png : 交叉後の辺ごとのCO2排出量

- Proposed_mutation.py : 突然変異の前後におけるルートおよび辺ごとのCO2排出量を出力
  - mutation_route_1.png : 突然変異前のルート
  - mutation_route_2.png : 突然変異後のルート
  - mutation_CO2_1.png : 突然変異前の辺ごとのCO2排出量
  - mutation_CO2_2.png : 突然変異後の辺ごとのCO2排出量

- Proposed_pm.py : 突然変異確率ごとの近似解を出力

- Proposed_pm.py : 淘汰の前後における母集団を出力
  - selection_1.png : 交叉および突然変異前の母集団
  - selection_2.png : 淘汰前の母集団
  - selection_3.png : 淘汰後の母集団
