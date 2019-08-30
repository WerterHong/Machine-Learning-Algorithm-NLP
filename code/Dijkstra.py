# 找出开销最低的节点
# python
def find_lowest_cost_node(costs, processed):
  lowest_cost = float("inf")
  lowest_cost_node  =None
  for node in costs:
    cost = costs[node]
    if cost < lowest_cost and node not in processed:
      lowest_cost = cost
      lowest_cost_node = node
  return lowest_cost_node

def dijkstra(costs, processed):
  node = find_lowest_cost_node(costs, processed)
  while node is not None:
    cost = costs[node]
    neighbors = graph[node]
    for n in neighbors.keys():
      new_cost = cost + neighbors[n]
      if costs[n] > new_cost:
        costs[n] = new_cost
        parents[n] = node
    processed.append(node)
    node = find_lowest_cost_node(costs, processed)

if __name__ == '__main__':
  graph = {}
  graph["Start"] = {}
  graph["Start"]["A"] = 10
  graph["Start"]["B"] = 12
  graph["A"] = {}
  graph["A"]["B"] = 9
  graph["A"]["E"] = 8
  graph["B"] = {}
  graph["B"]["C"] = 1
  graph["B"]["D"] = 3
  graph["C"] = {}
  graph["C"]["D"] = 3
  graph["C"]["F"] = 6
  graph["D"] = {}
  graph["D"]["E"] = 7
  graph["E"] = {}
  graph["E"]["F"] = 5
  graph["E"]["G"] = 8
  graph["F"] = {}
  graph["F"]["G"] = 9
  graph["F"]["Final"] = 11
  graph["G"] = {}
  graph["G"]["Final"] = 2
  graph["Final"] = {}

  # 存储开销表
  infinity = float("inf")
  costs = {}
  costs["A"] = 10
  costs["B"] = 12
  costs["C"] = infinity
  costs["D"] = infinity
  costs["E"] = infinity
  costs["F"] = infinity
  costs["G"] = infinity
  costs["Final"] = infinity

  # 存储父节点的散列表
  parents = {}
  parents["A"] = "Start"
  parents["B"] = "Start"
  parents["Final"] = None

  # parents用于记录处理过的节点
  processed = []

  dijkstra(costs, processed)

  print (parents)
