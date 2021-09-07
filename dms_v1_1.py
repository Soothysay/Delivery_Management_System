# -*- coding: utf-8 -*-
"""DMS_V1.1.ipynb

"""# Importing Required Libraries"""

import random
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

"""# Haversine Distance Computation Formula in Kms"""

# Haversine Distance Taken. We can take other distance as well
def distance(loc1,loc2):
   from sklearn.metrics.pairwise import haversine_distances
   from math import radians
   loc1_in_radians=[radians(_) for _ in loc1]
   loc2_in_radians=[radians(_) for _ in loc2]
   result = haversine_distances([loc1_in_radians, loc2_in_radians])
   result=result * 6371000/1000 # Converting the distance into kilometers
   return (float(result[0,1]))

"""# Creating Dummy Data"""

def dummy_data_creator():
  seed = 10101
  random.seed(seed)
  num_customers = 300 # Total Numbrer of customers ordering
  num_candidates = 50 # Total number of riders
  customer_locs_global=[(random.random()-0.5, random.random()-0.5)
                for i in range(num_customers)] # Getting Latitude and Longitude of customers
  candidate_locs_global=[(random.random()-0.5, random.random()-0.5)
                for i in range(num_candidates)] # Getting Latitude and Longitude of riders
  shoploc_global=[(random.random()-0.5, random.random()-0.5)
                for i in range(25)] # Getting latitude and longitude of shops
  orderto_global=[(i,random.randint(0,24))for i in range(num_customers)] # Getting a map of which customer orders from which store
  # Creating Individual Databases
  shop=pd.DataFrame()
  order=pd.DataFrame()
  rider=pd.DataFrame()
  shop['Name']=['Shop'+str(i+1) for i in range (25)]
  shop['Shop Latitude-Longitude']=shoploc_global
  shop['Index']=[i for i in range(25)]
  order['Customer Name']=['Customer'+str(i+1) for i in range (num_customers)]
  order['Customer Latitude-Longitude']=customer_locs_global
  order['Index']=[orderto_global[i][1] for i in range(len(orderto_global))]
  #customer['Index']=[i for i in range(num_customers)]
  rider['Name']=['Rider'+str(i+1) for i in range (num_candidates)]
  rider['Rider Latitude-Longitude']=candidate_locs_global
  #rider['Index']=[i for i in range(num_candidates)]
  order1=pd.merge(order,shop,how='inner',on='Index')
  dist=[]
  for i in range(len(order1)):
    dist.append(distance(order1['Customer Latitude-Longitude'].loc[i],order1['Shop Latitude-Longitude'].loc[i]))
  order1['Distance']=dist
  rider['Number of Orders in Queue']=[random.randint(0,5) for i in range (num_candidates)]
  rider.to_csv('Total_Riders.csv',index=False)
  order1.to_csv('Orders.csv',index=False)
  return rider,order,order1

"""# Rider Allocation using Optimization"""

def optimize(rider,order1,maxorders=3):
  costs=[]
  for i in range(len(rider)):
    riderdist=[]
    riderloc=rider['Rider Latitude-Longitude'].loc[i]
    riderqueue=rider['Number of Orders in Queue'].loc[i]
    for j in range(len(order1)):
      shoploc=order1['Shop Latitude-Longitude'].loc[j]
      totaldist=(float)((riderqueue+1)*distance(riderloc,shoploc)) # Minor Change Made here
      riderdist.append(totaldist)
    costs.append(riderdist)
  num_workers = len(costs)
  num_tasks = len(costs[0])
  # Solver
  # Create the mip solver with the SCIP backend.
  solver = pywraplp.Solver.CreateSolver('SCIP')
  # Variables
  # x[i, j] is an array of 0-1 variables, which will be 1
  # if worker i is assigned to task j.
  x = {}
  for i in range(num_workers):
    for j in range(num_tasks):
      x[i, j] = solver.IntVar(0, 1, '')

  # Constraints
  # Each worker is assigned to at most 3 tasks.
  for i in range(num_workers):
    solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) <= maxorders)

  # Each task is assigned to exactly one worker.
  for j in range(num_tasks):
    solver.Add(solver.Sum([x[i, j] for i in range(num_workers)]) == 1)

  # Objective
  objective_terms = []
  for i in range(num_workers):
    for j in range(num_tasks):
      objective_terms.append(costs[i][j] * x[i, j])
  solver.Minimize(solver.Sum(objective_terms))
  # Solve
  status = solver.Solve()
  customerassigned=[]
  riderassigned=[]
  optimized=pd.DataFrame()
  if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
    print('Total Distance = ', solver.Objective().Value(), '\n')
    for i in range(num_workers):
      for j in range(num_tasks):
        # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
        if x[i, j].solution_value() > 0.5:
          #print('Worker %d assigned to task %d.  Cost = %d' %(i, j, costs[i][j]))
          customerassigned.append(order1['Customer Name'].loc[j])
          riderassigned.append(rider['Name'].loc[i])
    optimized['Rider']=riderassigned
    optimized['Customer']=customerassigned
    return optimized
  else:
    print('Solver Could not solve')

"""# Caller Function"""

def call():
  rider,order,order1=dummy_data_creator()
  print('Rider Table')
  display(rider)
  print('Order Table')
  display(order1)
  rider_available=rider[rider['Number of Orders in Queue']<4]
  rider_available.reset_index(inplace=True,drop=True)
  print('Available riders')
  display(rider_available)
  rider_available.to_csv('Riders_Available.csv',index=False)
  maxorders=4 # Maximum Number of orders assigned to each driver
  if maxorders*len(rider_available)<=len(order1):
    selected_orders=order1.head((maxorders)*len(rider_available))
    unselected_orders=order1.tail(len(order1)-((maxorders)*len(rider_available)))
  else:
    selected_orders=order1
    unselected_orders='nil'
  print('Selected Orders Length:')
  print(len(selected_orders))
  print('Unselected Orders:')
  print(unselected_orders)
  #print(len(selected_orders))
  optimized_data=optimize(rider_available,selected_orders,maxorders)
  print('Assignment Table')
  display(optimized_data)
  optimized_data.to_csv('Optimized_Rider_Allocation.csv',index=False)
  table=pd.DataFrame()
  table['Rider']=optimized_data['Rider'].value_counts().keys().tolist()
  table['Count']=optimized_data['Rider'].value_counts().tolist()
  display(table)
  for i in range(len(table)):
    for j in range(len(rider)):
      if (table['Rider'].loc[i]==rider['Name'].loc[j]):
        p=rider['Number of Orders in Queue'].loc[j]
        q=table['Count'].loc[i]
        rider['Number of Orders in Queue'].loc[j]=p+q
  print('Updated Rider Table:')
  display(rider)
  rider.to_csv('Updated_Riders_Table.csv',index=False)
  return optimized_data,order1,rider

assign,order,rider=call()

"""# Task Assign Order"""

def task_order_assign(riderindex,assign,order,rider):
  df1=assign.loc[assign['Rider']==riderindex,'Customer'].tolist()
  riderloc=rider.loc[rider['Name']==riderindex,'Rider Latitude-Longitude'].tolist()[0]
  distanced=pd.DataFrame()
  dist=[]
  cust=[]
  shops=[]
  for i in range(len(df1)):
    dist.append(distance(order.loc[order['Customer Name']==df1[i],'Shop Latitude-Longitude'].tolist()[0],riderloc))
    shops.append(order.loc[order['Customer Name']==df1[i],'Name'].tolist()[0])
    cust.append(df1[i])
  distanced['Shop']=shops
  distanced['Customer']=cust
  distanced['Distance']=dist
  distances=distanced.sort_values('Distance')
  distances.reset_index(inplace=True,drop=True)
  #lastloc=order.loc[order['Name']==distances['Shop'].loc[len(distances)-1],'Shop Latitude-Longitude'].tolist()[0]
  pickupqueue=distances['Customer'].tolist()
  dropoffqueue=distanced['Customer'].tolist()
  print('Pick Up Queue')
  print(pickupqueue)
  print('Drop Off Queue')
  print(dropoffqueue)
  #print(distances)

rider_assigned=assign['Rider'].unique().tolist()
for i in range(len(rider_assigned)):
  print(rider_assigned[i])
  task_order_assign(rider_assigned[i],assign,order,rider)

"""# Assignment of Route using TSP Optimization"""

import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def task_order_assign_tsp(riderindex,assign,order,rider):
  df1=assign.loc[assign['Rider']==riderindex,'Customer'].tolist()
  riderloc=rider.loc[rider['Name']==riderindex,'Rider Latitude-Longitude'].tolist()[0]
  distanced=pd.DataFrame()
  dist=[]
  cust=[]
  shops=[]
  for i in range(len(df1)):
    dist.append(distance(order.loc[order['Customer Name']==df1[i],'Shop Latitude-Longitude'].tolist()[0],riderloc))
    shops.append(order.loc[order['Customer Name']==df1[i],'Name'].tolist()[0])
    cust.append(df1[i])
  distanced['Shop']=shops
  distanced['Customer']=cust
  distanced['Distance']=dist
  distances=distanced.sort_values('Distance')
  distances.reset_index(inplace=True,drop=True)
  lastloc=order.loc[order['Name']==distances['Shop'].loc[len(distances)-1],'Shop Latitude-Longitude'].tolist()[0]
  pickupqueue=distances['Customer'].tolist()
  dropoffqueue=[]
  places=['Driver Location']
  latlongs=[lastloc]
  for i in range(len(df1)):
    places.append(df1[i])
    latlongs.append(order.loc[order['Customer Name']==df1[i],'Customer Latitude-Longitude'].tolist()[0])
  distance_matrix=[]
  for i in range(len(latlongs)):
    deldist=[]
    for j in range(len(latlongs)):
      totaldist=(float)(distance(latlongs[i],latlongs[j]))
      deldist.append(totaldist)
    distance_matrix.append(deldist)
  # Instantiate the data problem.
  data = {}
  data['distance_matrix']=distance_matrix
  data['num_vehicles'] = 1
  data['depot'] = 0
  # Create the routing index manager.
  manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),data['num_vehicles'], data['depot'])
  # Create Routing Model.
  routing = pywrapcp.RoutingModel(manager)
  def distance_callback(from_index, to_index):
    """Returns the distance between the two nodes."""
    # Convert from routing variable Index to distance matrix NodeIndex.
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return distance_matrix[from_node][to_node]
  transit_callback_index = routing.RegisterTransitCallback(distance_callback)
  # Define cost of each arc.
  routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
  search_parameters = pywrapcp.DefaultRoutingSearchParameters()
  search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
  search_parameters.time_limit.seconds = 3 # Can be changed
  search_parameters.log_search = True
  # Solve the problem.
  solution = routing.SolveWithParameters(search_parameters)
  index = routing.Start(0)
  route_distance = 0
  while not routing.IsEnd(index):
    dropoffqueue.append(places[manager.IndexToNode(index)])
    previous_index = index
    index = solution.Value(routing.NextVar(index))
    route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
  #print('Pick Up Queue')
  #print(pickupqueue)
  #print('Drop Off Queue')
  dropoffqueue.remove('Driver Location')
  #print(dropoffqueue)
  return pickupqueue,dropoffqueue,route_distance

rider_assigned=assign['Rider'].unique().tolist()
pick=[]
drop=[]
tot=0.0
for i in range(len(rider_assigned)):
  #print(rider_assigned[i])
  p,d,dist=task_order_assign_tsp(rider_assigned[i],assign,order,rider)
  pick.append(p)
  drop.append(d)
  tot=tot+dist
routings=pd.DataFrame()
routings['Rider']=rider_assigned
routings['Pickup Order']=pick
routings['Dropoff Order']=drop
routings.to_csv('routes.csv',index=False)
print(tot)