import array
import json
import math
import random
from time import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from deap import base, creator, tools

with open("data/E-n23-k3.json", "r") as gvrp_data:
    gvrp = json.load(gvrp_data)

n = gvrp["DIMENSION"]
m = gvrp["VEHICLES"]
Q = gvrp["CAPACITY"]
pos_list = gvrp["NODE_COORD_SECTION"]
demand_list = gvrp["DEMAND_SECTION"]

INF = 2 ** 30

# 移動距離とCO2排出量を最小化
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)


# 初期配列を生成(ランダム)
def init_pop():
    init_list = [0] * (m-1)
    for i in range(n-1):
        init_list.append(i+1)
    random.shuffle(init_list)
    init_list.insert(0, 0)
    init_list.append(0)
    return init_list


# 遺伝子をルートごとに分ける
def split(individual):
    route_list = []
    route = []
    for i in range(len(individual)):
        ith = individual[i]
        route.append(ith)
        if route == [0, 0]:
            route = [0]
        if i and ith == 0:
            if route != [0]:
                route_list.append(route)
                route = [0]
    return route_list


# ルートが制約条件を満たしているか判定
def is_route(individual):
    route_list = split(individual)
    load = 0
    for route in route_list:
        for i in range(len(route)):
            ith = route[i]
            load += demand_list[ith]
            if load > Q:
                return False
    return True


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
            try:
                weight_list[r] += demand_list[ith]
            except:
                print(r, ith)
                print(len(weight_list), len(demand_list))
                exit(-1)

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


# 突然変異(2点の入れ替え)
def mutation(individual):
    route_list = split(individual)
    r1_1 = random.randint(0, len(route_list)-1)
    r2_1 = random.randint(0, len(route_list)-1)
    while len(route_list) > 1 and r1_1 == r2_1:
        r2_1 = random.randint(0, len(route_list)-1)
    r1_2 = random.randint(0, len(route_list[r1_1])-1)
    r2_2 = random.randint(0, len(route_list[r2_1])-1)
    g1 = route_list[r1_1][r1_2]
    g2 = route_list[r2_1][r2_2]
    i1 = individual.index(g1)
    i2 = individual.index(g2)
    individual[i1] = g2
    individual[i2] = g1
    return individual


# 各ルートを可視化
def vis_route(ind, path):
    fig = plt.figure()
    G = nx.DiGraph()
    G.add_nodes_from([i for i in range(n)])
    pos = {i: pos_list[i] for i in range(n)}
    route_list = split(ind)
    color_list = ["red", "green", "blue", "orange", "purple"]
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


# 辺ごとのCO2排出量を可視化
def vis_CO2(ind, path):
    fig = plt.figure()
    G = nx.DiGraph()
    G.add_nodes_from([i for i in range(n)])
    pos = {i: pos_list[i] for i in range(n)}
    route_list = split(ind)
    CO2_list = get_CO2_list(ind)
    print(CO2_list)
    color_list = ["moccasin", "yellow", "orange", "hotpink", "red"]
    node_color = ["black"] * n
    cnt = 0
    print(route_list)
    for i, route in enumerate(route_list):
        for j in range(len(route)-1):
            idx = min(4, max(0, int(CO2_list[cnt] / 4) - 1))
            G.add_edge(route[j], route[j+1], color=color_list[idx], weight=1+idx)
            cnt += 1
    edge_color = [G[u][v]["color"] for u,v in G.edges]
    edge_weights = [G[u][v]["weight"] for u,v in G.edges]
    nx.draw_networkx(G, pos=pos, with_labels=True, font_weight="bold", edge_color=edge_color, node_color=node_color, width=edge_weights, node_size=20, font_size=0)
    fig.savefig(path)


# 各関数の設定
toolbox = base.Toolbox()
toolbox.register("individual_random", tools.initIterate, creator.Individual, init_pop)  # ランダム
toolbox.register("population_random", tools.initRepeat, list, toolbox.individual_random)
toolbox.register("select", tools.selTournament, tournsize=2)  # 選択(二分トーナメント選択)
toolbox.register("mate", tools.cxPartialyMatched)  # 交叉(部分写像交叉)
toolbox.register("mutate", mutation)  # 突然変異
toolbox.register("evaluate", evaluation)  # 評価


def main():
    t1 = time()
    random.seed(1)
    N = 100  # 集団の個体数
    CXPB = 0.25  # 交叉確率
    MUPB = 0.35  # 突然変異確率

    # 各世代の最小値、最大値を出力
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "min", "max"

    # 第一世代の生成
    pop = toolbox.population_random(n)  # その他はランダム挿入
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, evals=N, **record)

    # 各世代の処理
    NGEN = 100
    best_ind = pop[0]
    best_obj = (INF, INF)
    best_cost = INF
    for gen in range(NGEN):
        # 二分トーナメント選択で２つの親を選択
        offspring = [toolbox.clone(ind) for ind in pop]
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            # 部分写像交叉
            if random.random() <= CXPB:
                ind1, ind2 = toolbox.mate(ind1, ind2)

            # 確率 pm で突然変異
            if random.random() <= MUPB:
                ind1c = toolbox.mutate(ind1)
                if is_route(ind1c):
                    ind1 = ind1c
            if random.random() <= MUPB:
                ind2c = toolbox.mutate(ind2)
                if is_route(ind2c):
                    ind2 = ind2c

        pop = toolbox.select(pop, k=N)  # 高速非優越ソート
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=N, **record)
        print(logbook.stream)

    best_ind = pop[0]
    best_obj = pop[0].fitness.values
    best_cost = INF
    for ind in pop:
        ind_cost = ind.fitness.values[0]**2 + ind.fitness.values[1]**2
        if is_route(ind) and ind_cost < best_cost:
            best_ind = ind
            best_obj = ind.fitness.values
            best_cost = ind_cost
    if best_cost == INF:
        print(best_ind)
        print(split(best_ind))
        print("車両数の制約条件を満たさない。")
        exit(1)

    print(best_ind)
    print(f"移動距離：{round(best_obj[0], 2)}km")
    print(f"CO2排出量：{round(best_obj[1], 2)}kg")
    t2 = time()
    print(f"実行時間：{round(t2 - t1, 2)}秒")

    # 各ルートを可視化
    vis_route(best_ind, "figure/route_Jemai.png")
    # 辺ごとのCO2排出量を可視化
    vis_CO2(best_ind, "figure/CO2_Jemai.png")


if __name__ == "__main__":
    main()
