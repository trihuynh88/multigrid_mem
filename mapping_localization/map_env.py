import numpy as np
from numpy.random import random_integers as rand
from constants import MAP_SIZE
from constants import MAP_COMPLEXITY
from constants import MAP_DENSITY
from constants import BLANK_MAP
from constants import INPUT_SIZE
import pdb

class MapEnv(object):
  def __init__(self,batch_size,generic_map_size):
    self.batch_size = batch_size
    self.agent_mark = 4
    self.unknown_mark = 3
    self.generic_map_size = generic_map_size
    self.unavail_mark = 1
    self.free_mark = 0
    self.visited_mark = 2
    self.radius = int(INPUT_SIZE/2)

  def create_blank_map(self,size):
    blank_map = np.ones((size,size),dtype=int)*self.free_mark
    for i in range(size):
      blank_map[0,i] = self.unavail_mark
      blank_map[size-1,i] = self.unavail_mark
      blank_map[i,0] = self.unavail_mark
      blank_map[i,size-1] = self.unavail_mark
    return blank_map

  def maze(self, width=13, height=13, complexity=.1, density=.75):
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    complexity = int(complexity * (5 * (shape[0] + shape[1]))) 
    density    = int(density * ((shape[0] // 2) * (shape[1] // 2))) 
    Z = np.ones(shape, dtype=int)*self.free_mark
    Z[0, :] = Z[-1, :] = self.unavail_mark
    Z[:, 0] = Z[:, -1] = self.unavail_mark
    for i in range(density):
        x, y = rand(0, shape[1] // 2) * 2, rand(0, shape[0] // 2) * 2
        Z[y, x] = self.unavail_mark
        for j in range(complexity):
            neighbours = []
            if x > 1:             neighbours.append((y, x - 2))
            if x < shape[1] - 2:  neighbours.append((y, x + 2))
            if y > 1:             neighbours.append((y - 2, x))
            if y < shape[0] - 2:  neighbours.append((y + 2, x))
            if len(neighbours):
                y_,x_ = neighbours[rand(0, len(neighbours) - 1)]
                if Z[y_, x_] == self.free_mark:
                    Z[y_, x_] = self.unavail_mark
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = self.unavail_mark
                    x, y = x_, y_
    return Z

  def clip_bound(self,val):
    if val<0:
      return 0
    if val>=MAP_SIZE:
      return MAP_SIZE-1
    return val    

  def to_tuple(self,arr):
    return tuple([tuple(l) for l in arr.tolist()])

  def reset_agent_pos(self):
    self.init_map = np.zeros((self.batch_size,MAP_SIZE,MAP_SIZE),dtype=int)
    if BLANK_MAP:
      for batch_ind in range(self.batch_size):
        self.init_map[batch_ind,:,:] = self.create_blank_map(MAP_SIZE)
    else:
      for batch_ind in range(self.batch_size):
        self.init_map[batch_ind,:,:] = self.maze(MAP_SIZE,MAP_SIZE,MAP_COMPLEXITY,MAP_DENSITY)
    self.maps = np.ones((self.batch_size,self.init_map.shape[1],self.init_map.shape[2]),dtype=int)*self.unknown_mark
    self.generic_maps = np.ones((self.batch_size,self.generic_map_size,self.generic_map_size),dtype=int)*self.unknown_mark
    self.agent_pos = np.zeros((self.batch_size,2),dtype=int)
    self.agent_pos[:,0] = int(MAP_SIZE/2)
    self.agent_pos[:,1] = int(MAP_SIZE/2)
    self.pattern_loc = []

    for batch_ind in range(self.batch_size):
      ystart = self.clip_bound(self.agent_pos[batch_ind][0]-self.radius)
      yend = self.clip_bound(self.agent_pos[batch_ind][0]+self.radius)
      xstart = self.clip_bound(self.agent_pos[batch_ind][1]-self.radius)
      xend = self.clip_bound(self.agent_pos[batch_ind][1]+self.radius)
      y,x = np.ogrid[ystart:yend+1,xstart:xend+1]
      self.maps[batch_ind,y,x] = self.init_map[batch_ind,y,x]
      if self.maps[batch_ind][self.agent_pos[batch_ind,0]][self.agent_pos[batch_ind,1]] == self.free_mark:
        self.maps[batch_ind][self.agent_pos[batch_ind,0]][self.agent_pos[batch_ind,1]] = self.visited_mark

      old_ego = self.maps[batch_ind,y,x]
      generic_pos = (self.generic_map_size/2,self.generic_map_size/2)

      self.pattern_loc.append({})
      self.pattern_loc[batch_ind][self.to_tuple(self.init_map[batch_ind,y,x])] = np.zeros((48,48),dtype=int)
      self.pattern_loc[batch_ind][self.to_tuple(self.init_map[batch_ind,y,x])][generic_pos[0],generic_pos[1]] = 1

      ystart = generic_pos[0]-self.radius
      yend = generic_pos[0]+self.radius
      xstart = generic_pos[1]-self.radius
      xend = generic_pos[1]+self.radius
      y,x = np.ogrid[ystart:yend+1,xstart:xend+1]
      self.generic_maps[batch_ind,y,x] = np.copy(old_ego)
     
    self.generic_pos = np.ones((self.batch_size,2),dtype=int)*(self.generic_map_size/2)      

  def move_spiral(self):
    r0 = int(MAP_SIZE/2)
    c0 = int(MAP_SIZE/2)
    r_cur = self.agent_pos[0,0]
    c_cur = self.agent_pos[0,1]
    rdelta = r_cur-r0
    cdelta = c_cur-c0

    r_next = r_cur
    c_next = c_cur

    if cdelta>=0:
      rlow = r0 + (-cdelta+1)
      rhigh = r0 + (cdelta-1)
      if r_cur<rlow:
        r_next = r_cur
        c_next += 1
      elif r_cur<=rhigh:
        r_next += 1
        c_next = c_cur
      else:
        r_next = r_cur
        c_next -= 1

    else:
      rlow = r0 + (cdelta+1)
      rhigh = r0 + (-cdelta)
      if r_cur<rlow:
        r_next = r_cur
        c_next += 1
      elif r_cur<=rhigh:
        r_next -= 1
        c_next = c_cur
      else:
        r_next = r_cur
        c_next -= 1

    if r_next<1 or r_next>=MAP_SIZE-1 or c_next<1 or c_next>=MAP_SIZE-1:
      return True
    else:
      self.agent_pos[:,0] = r_next
      self.agent_pos[:,1] = c_next
      self.generic_pos[:,0] += (r_next-r_cur)
      self.generic_pos[:,1] += (c_next-c_cur) 

      ystart = self.clip_bound(self.agent_pos[0,0]-self.radius)
      yend = self.clip_bound(self.agent_pos[0,0]+self.radius)
      xstart = self.clip_bound(self.agent_pos[0,1]-self.radius)
      xend = self.clip_bound(self.agent_pos[0,1]+self.radius)

      y,x = np.ogrid[ystart:yend+1,xstart:xend+1]

      ystart_generic = self.generic_pos[0,0]-self.radius
      yend_generic = self.generic_pos[0,0]+self.radius
      xstart_generic = self.generic_pos[0,1]-self.radius
      xend_generic = self.generic_pos[0,1]+self.radius
      y_generic,x_generic = np.ogrid[ystart_generic:yend_generic+1,xstart_generic:xend_generic+1]

      for batch_ind in range(self.batch_size):
        if self.maps[batch_ind][self.agent_pos[batch_ind,0]][self.agent_pos[batch_ind,1]] == self.free_mark:
          self.maps[batch_ind][self.agent_pos[batch_ind,0]][self.agent_pos[batch_ind,1]] = self.visited_mark
        if self.generic_maps[batch_ind][self.generic_pos[batch_ind,0]][self.generic_pos[batch_ind,1]] == self.free_mark:
          self.generic_maps[batch_ind][self.generic_pos[batch_ind,0]][self.generic_pos[batch_ind,1]] = self.visited_mark
        old_ego = self.maps[batch_ind,y,x]
        new_ego = self.init_map[batch_ind,y,x]
        indices = np.where(old_ego == self.unknown_mark)
        old_ego[indices[0],indices[1]] = new_ego[indices[0],indices[1]]
        self.maps[batch_ind,y,x] = old_ego
        self.generic_maps[batch_ind,y_generic,x_generic] = old_ego
        
        cur_key = self.to_tuple(self.init_map[batch_ind,y,x])
        if cur_key in self.pattern_loc[batch_ind]:
          self.pattern_loc[batch_ind][cur_key][self.generic_pos[batch_ind,0],self.generic_pos[batch_ind,1]] = 1
        else:
          self.pattern_loc[batch_ind][cur_key] = np.zeros((48,48),dtype=int)
          self.pattern_loc[batch_ind][cur_key][self.generic_pos[batch_ind,0],self.generic_pos[batch_ind,1]] = 1

      return False
      
  def get_input_ego(self):
    input_ego = np.zeros((self.batch_size,INPUT_SIZE,INPUT_SIZE),dtype=int)
    offset = (INPUT_SIZE-3)/2
    radius = (INPUT_SIZE-1)/2
    newmap = np.ones((self.batch_size,self.init_map[0].shape[0]+offset*2,self.init_map[0].shape[1]+offset*2),dtype=int)*self.unknown_mark
    newmap[:,offset:newmap.shape[1]-offset,offset:newmap.shape[2]-offset] = np.copy(self.init_map[:,:,:])
    for i in range(self.batch_size):
      ystart = self.agent_pos[i,0]+offset-radius
      yend = self.agent_pos[i,0]+offset+radius
      xstart = self.agent_pos[i,1]+offset-radius
      xend = self.agent_pos[i,1]+offset+radius
      y,x = np.ogrid[ystart:yend+1,xstart:xend+1]
      input_ego[i] = newmap[i,y,x]
    return input_ego
  

