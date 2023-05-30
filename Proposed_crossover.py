import array
import json
import math
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from deap import base, creator, tools

with open("data/E-n51-k5.json", "r") as gvrp_data:
    gvrp = json.load(gvrp_data)

n = gvrp["DIMENSION"]
Q = gvrp["CAPACITY"]
pos_list = gvrp["NODE_COORD_SECTION"]
demand_list = gvrp["DEMAND_SECTION"]

INF = 2 ** 30

# 移動距離とCO2排出量を最小化
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", array.array, typecode="i", fitness=creator.FitnessMin)


# 初期配列を生成(最近傍探索)
def init_pop():
    nodef = 0
    init_list = []
    used_list = [False] * n
    used_list[0] = True
    for i in range(n-1):
        fx = pos_list[nodef][0]
        fy = pos_list[nodef][1]
        dis_min = INF
        nodet = 0
        for node in range(1, n):
            if used_list[node]:
                continue
            tx = pos_list[node][0]
            ty = pos_list[node][1]
            dis_ft = math.sqrt((fx - tx) * (fx - tx) + (fy - ty) * (fy - ty))
            if dis_ft < dis_min:
                nodet = node
                dis_min = dis_ft
        init_list.append(nodet)
        used_list[nodet] = True
        nodef = nodet
    return init_list


# 遺伝子をルートごとに分ける
def split(individual):
    route_list = []
    route = [0]
    load = 0
    for i in range(n-1):
        ith = individual[i]
        load_i = demand_list[ith]
        if load + load_i <= Q:
            load += load_i
            route.append(ith)
        else:
            route.append(0)
            route_list.append(route)
            load = load_i
            route = [0, ith]
    route.append(0)
    route_list.append(route)
    return route_list


# 2頂点間におけるCO2排出量を求める
def CO2emission(d, q):
    return d * ((0.749 - 0.546) / Q * q + 0.546)


# 評価
def evaluation(individual):
    route_list = split(individual)

    # 各ルートで運ぶ荷物の重さを求める
    weight_list = [0] * len(route_list)
    for r in range(len(route_list)):
        for ith in route_list[r]:
            if ith == 0:
                continue
            weight_list[r] += demand_list[ith]

    # (fx, fy)から(tx, ty)に移動する際の移動距離、CO2排出量を求める
    fit_dis = 0
    fit_CO2 = 0
    for r in range(len(route_list)):
        fx = pos_list[0][0]
        fy = pos_list[0][1]
        for ith in route_list[r]:
            tx = pos_list[ith][0]
            ty = pos_list[ith][1]
            dis = math.sqrt((fx - tx) * (fx - tx) + (fy - ty) * (fy - ty))
            fit_dis += dis
            fit_CO2 += CO2emission(dis, weight_list[r])
            if ith != 0:
                weight_list[r] -= demand_list[ith]
            fx = tx
            fy = ty
    return fit_dis, fit_CO2


# 各ルートを可視化
def vis_route(ind, path):
    fig = plt.figure()
    G = nx.DiGraph()
    G.add_nodes_from([i for i in range(n)])
    pos = {i: pos_list[i] for i in range(n)}
    route_list = split(ind)
    color_list = ["red", "green", "blue", "orange", "purple", "purple", "purple"]
    edge_list = []
    node_color = ["black"] * n
    for i, route in enumerate(route_list):
        color = color_list[i]
        for j in range(len(route)-1):
            edge = (route[j], route[j+1], {"color": color})
            edge_list.append(edge)
            node_color[route[j+1]] = color
    G.add_edges_from(edge_list)
    edge_color = [edge["color"] for edge in G.edges.values()]
    nx.draw_networkx(G, pos=pos, with_labels=True, font_weight='bold', edge_color=edge_color, node_color=node_color, node_size=20, font_size=0)
    fig.savefig(path)


def get_CO2_list(individual):
    route_list = split(individual)

    # 各ルートで運ぶ荷物の重さを求める
    weight_list = [0] * len(route_list)
    for r in range(len(route_list)):
        for ith in route_list[r]:
            if ith == 0:
                continue
            weight_list[r] += demand_list[ith]

    # (fx, fy)から(tx, ty)に移動する際の移動距離、CO2排出量を求める
    CO2_list = []
    for r in range(len(route_list)):
        fx = pos_list[0][0]
        fy = pos_list[0][1]
        for ith in route_list[r]:
            tx = pos_list[ith][0]
            ty = pos_list[ith][1]
            dis = math.sqrt((fx - tx) * (fx - tx) + (fy - ty) * (fy - ty))
            CO2_ith = CO2emission(dis, weight_list[r])
            if CO2_ith > 0.1:
                CO2_list.append(CO2_ith)
            if ith != 0:
                weight_list[r] -= demand_list[ith]
            fx = tx
            fy = ty
    return CO2_list


# 突然変異(2opt法)
def mutation(individual):
    while True:
        cnt = 0
        for i in range(n-2):
            for j in range(n-2):
                individual2 = deepcopy(individual)
                individual2[i], individual2[j] = individual2[j], individual2[i]
                if evaluation(individual2) < evaluation(individual):
                    individual = individual2
                    cnt += 1
        if cnt == 0:
            break
    return individual


# ind_cが任意の親と重複していないか確認
def is_duplicate(ind_c, pop):
    for ind in pop:
        if ind_c == ind:
            return True
    return False


# 辺ごとのCO2排出量を可視化
def vis_CO2(ind, path):
    fig = plt.figure()
    G = nx.DiGraph()
    G.add_nodes_from([i for i in range(n)])
    pos = {i: pos_list[i] for i in range(n)}
    route_list = split(ind)
    CO2_list = get_CO2_list(ind)
    color_list = ["moccasin", "yellow", "orange", "hotpink", "red"]
    node_color = ["black"] * n
    cnt = 0
    for i, route in enumerate(route_list):
        for j in range(len(route)-1):
            idx = max(0, int(min(19, CO2_list[cnt]) / 4) - 1)
            G.add_edge(route[j], route[j+1], color=color_list[idx], weight=1+idx)
            cnt += 1
    edge_color = [G[u][v]["color"] for u,v in G.edges]
    edge_weights = [G[u][v]["weight"] for u,v in G.edges]
    nx.draw_networkx(G, pos=pos, with_labels=True, font_weight="bold", edge_color=edge_color, node_color=node_color, width=edge_weights, node_size=20, font_size=0)
    fig.savefig(path)


# 各関数の設定
toolbox = base.Toolbox()
toolbox.register("individual_nearest", tools.initIterate, creator.Individual, init_pop)  # 最近傍探索
toolbox.register("population_nearest", tools.initRepeat, list, toolbox.individual_nearest)
toolbox.register("indices", random.sample, range(1, n), n-1)
toolbox.register("individual_random", tools.initIterate, creator.Individual, toolbox.indices)  # ランダム挿入
toolbox.register("population_random", tools.initRepeat, list, toolbox.individual_random)
toolbox.register("select", tools.selNSGA2)  # 選択(高速非優越ソート)
toolbox.register("mate", tools.cxOrdered)  # 交叉(順序交叉)
toolbox.register("mutate", mutation)  # 突然変異
toolbox.register("evaluate", evaluation)  # 評価


def main():
    random.seed(1)
    NGEN = 1000
    N = 80  # 集団の個体数
    MUPB = 0.003  # 突然変異確率

    # 各世代の最小値、最大値を出力
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "min", "max"

    # 第一世代の生成
    pop = toolbox.population_nearest(n=3)  # 3つの個体は最近傍探索
    pop2 = toolbox.population_random(n=N-3)  # その他はランダム挿入
    pop = pop + pop2
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    pop = tools.selNSGA2(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=N, **record)

    # 各世代の処理
    for gen in range(1, NGEN):
        offspring = [toolbox.clone(ind) for ind in pop]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            ind1_p = toolbox.clone(ind1)
            ind2_p = toolbox.clone(ind2)
            eval1_1 = toolbox.evaluate(ind1_p)
            eval2_1 = toolbox.evaluate(ind2_p)

            # 順序交叉
            for i in range(len(ind1)):
                ind1[i] -= 1
                ind2[i] -= 1
            ind1, ind2 = toolbox.mate(ind1, ind2)
            for i in range(len(ind1)):
                ind1[i] += 1
                ind2[i] += 1

            eval1_2 = toolbox.evaluate(ind1)
            eval2_2 = toolbox.evaluate(ind2)

            # CO2の総排出量が条件を満たしたら出力
            if eval1_1[1] - eval1_2[1] > 20 and eval1_2[1] < 380:
                print(eval1_1)
                print(eval1_2)
                vis_route(ind1_p, "figure/crossover_route_1.png")
                vis_route(ind1, "figure/crossover_route_2.png")
                vis_CO2(ind1_p, "figure/crossover_CO2_1.png")
                vis_CO2(ind1, "figure/crossover_CO2_2.png")
                exit(0)
            if eval1_1[1] - eval1_2[1] > 20 and eval1_2[1] < 380:
                print(eval2_1)
                print(eval2_2)
                vis_route(ind1_p, "figure/crossover_route_1.png")
                vis_route(ind1, "figure/crossover_route_2.png")
                vis_CO2(ind1_p, "figure/crossover_CO2_1.png")
                vis_CO2(ind1, "figure/crossover_CO2_2.png")
                exit(0)

            # 確率 pm で突然変異
            if random.random() <= MUPB:
                ind1 = toolbox.mutate(ind1)
            if random.random() <= MUPB:
                ind2 = toolbox.mutate(ind2)

            # ind1が任意の親と重複していないか確認
            is_duplicate = False
            for ind in pop:
                if ind1 == ind:
                    is_duplicate = True
                    break
            if not is_duplicate:
                pop.append(ind1)
                ind1.fitness.values = toolbox.evaluate(ind1)

            # ind2が任意の親と重複していないか確認
            is_duplicate = False
            for ind in pop:
                if ind2 == ind:
                    is_duplicate = True
                    break
            if not is_duplicate:
                pop.append(ind2)
                ind2.fitness.values = toolbox.evaluate(ind2)

        pop = toolbox.select(pop, k=N)  # 高速非優越ソート
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=N, **record)
        print(logbook.stream)


if __name__ == "__main__":
    main()
