--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'
local doc = require 'argcheck.doc'

local Meter = torch.class('tnt.Meter', tnt)

doc[[
### tnt.Meter

When training a model, you generally would like to measure how the model is
performing. Specifically, you may want to measure the average processing time
required per batch of data, the classification error or AUC of a classifier a
validation set, or the precision@k of a retrieval model.

Meters provide a standardized way to measure a range of different measures,
which makes it easy to measure a wide range of properties of your models.

Nearly all meters (except `tnt.TimeMeter`) implement three methods:

   * `add()` which adds an observation to the meter.
   * `value()` which returns the value of the meter, taking into account all observations.
   * `reset()` which removes all previously added observations, resetting the meter.

The exact input arguments to the `add()` method vary depending on the meter.
Most meters define the method as `add(output, target)`, where `output` is the
output produced by the model and `target` is the ground-truth label of the data.

The `value()` method is parameterless for most meters, but for measures that
have a parameter (such as the k parameter in precision@k), they may take an
input argument.

An example of a typical usage of a meter is as follows:
```lua
local meter = tnt.<Measure>Meter()  -- initialize meter
for state, event in tnt.<Optimization>Engine:train{
   network   = network,
   criterion = criterion,
   iterator  = iterator,
} do
  if state == 'start-epoch' then
     meter:reset()  -- reset meter
  elseif state == 'forward-criterion' then
     meter:add(state.network.output, sample.target)  -- add value to meter
  elseif state == 'end-epoch' then
     print('value of meter:' .. meter:value())  -- get value of meter
  end
end
```
]]

Meter.__init = argcheck{
   {name="self", type="tnt.Meter"},
   call =
      function(self)
      end
}

Meter.reset = argcheck{
   {name="self", type="tnt.Meter"},
   call =
      function(self)
         error('A tnt.Meter should implement the reset() function.')
      end
}

Meter.value = argcheck{
   {name="self", type="tnt.Meter"},
   call =
      function(self)
         error('A tnt.Meter should implement the value() function.')
      end
}

Meter.add = argcheck{
   {name="self", type="tnt.Meter"},
   call =
      function(self)
         error('A tnt.Meter should implement the add() function.')
      end
}

Meter.__updatetopk = argcheck{
   {name='self',   type='tnt.Meter'},
   {name='output', type='torch.*Tensor'},
   {name='target', type='torch.*Tensor'},  -- target is k-hot vector
   {name='topk',   type='number'},         -- number of values to maintain
   {name='dim',    type='number',  default=1},    -- top-k selection dimension
   {name='desc',   type='boolean', default=true}, -- maintain largest values
   call = function(self, output, target, topk, dim, desc)
      assert(dim == 1 or dim == 2)

      -- make sure top-k buffer has the right size:
      local firstinput  = not (self.__topkoutput and self.__topktarget)
      self.__topkoutput = self.__topkoutput or output.new()
      self.__topktarget = self.__topktarget or target.new()
      self.__topkoutput:resize(output:size(1) + ((dim == 1) and topk or 0),
                               output:size(2) + ((dim == 2) and topk or 0))
      self.__topktarget:resize(target:size(1) + ((dim == 1) and topk or 0),
                               target:size(2) + ((dim == 2) and topk or 0))
      if firstinput then
         self.__topkoutput:fill(desc and -math.huge or math.huge)
      end

      -- copy new inputs into buffer:
      local otherdim = (dim == 1) and 2 or 1
      assert(output:size(otherdim) == self.__topkoutput:size(otherdim),
         string.format('incorrect size of dimension %d of output', otherdim))
      assert(target:size(otherdim) == self.__topktarget:size(otherdim),
         string.format('incorrect size of dimension %d of target', otherdim))
      self.__topkoutput:narrow(dim, topk + 1, output:size(dim)):copy(output)
      self.__topktarget:narrow(dim, topk + 1, target:size(dim)):copy(target)

      -- update top-k scores:
      local topoutput, topind = torch.topk(self.__topkoutput, topk, dim, desc)
      self.__topkoutput:narrow(dim, 1, topk):copy(topoutput)
      self.__topktarget:narrow(dim, 1, topk):copy(
         self.__topktarget:gather(dim, topind)
      )
   end
}
