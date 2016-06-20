--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local SplitDataset = torch.class('tnt.SplitDataset', 'tnt.Dataset', tnt)

SplitDataset.__init = argcheck{
   doc = [[
<a name="SplitDataset">
#### tnt.SplitDataset(@ARGP)
@ARGT

Partition a given `dataset`, according to the specified `partitions`.  Use
the method [select()](#SplitDataset.select) to select the current partition
in use.

The Lua hash table `partitions` is of the form (key, value) where key is a
user-chosen string naming the partition, and value is a number representing
the weight (in size) of the corresponding partition.

The sum of the partition weights may or may not sum to one
(`tnt.SplitDataset` will make them sum to one!).

Partionning is achieved linearly (no shuffling). See
[tnt.ShuffleDataset](#ShuffleDataset) if you want to shuffle the dataset
before partitioning.

Purpose: useful in machine learning to perform validation procedures.
]],
   {name='self', type='tnt.SplitDataset'},
   {name='dataset', type='tnt.Dataset'},
   {name='partitions', type='table'},
   call =
      function(self, dataset, partitions)

         -- create partition size tensor and table with partition names:
         self.__dataset = dataset
         local n = 0; for _,_ in pairs(partitions) do n = n + 1 end
         self.__partitionsizes = torch.DoubleTensor(n)
         self.__names = {}
         n = 0
         for key, val in pairs(partitions) do
            n = n + 1
            self.__partitionsizes[n] = val
            self.__names[key] = n
         end

         -- assertions:
         assert(
            self.__partitionsizes:nElement() >= 2,
            'SplitDataset should have at least two partitions'
         )
         assert(
            self.__partitionsizes:min() >= 0,
            'some partition sizes are negative'
         )
         assert(
            self.__partitionsizes:max() > 0,
            'all partitions are empty'
         )

         -- if partition sizes are fractions, convert to sizes:
         if math.abs(self.__partitionsizes:sum() - 1) < 1e-5 then
            self.__partitionsizes = self.__partitionsizes:double()
            self.__partitionsizes:mul(
               self.__dataset:size() / self.__partitionsizes:sum()
            ):floor():long()
         end

         -- select first partition by default:
         self.__partition = 1
      end
}

SplitDataset.select = argcheck{
   doc = [[
<a name="SplitDataset.select">
##### tnt.SplitDataset.select(@ARGP)
@ARGT

Switch the current partition in use to the one specified by `partition`,
which must be a string corresponding to one of the names provided at
construction.

The current dataset size changes accordingly, as well as the samples returned
by the `get()` method.
]],
   {name='self', type='tnt.SplitDataset'},
   {name='partition', type='string'},
   call =
      function(self, partition)
         self.__partition = self.__names[partition]
         if not self.__partition then error('partition not found') end
      end
}

SplitDataset.size = argcheck{
   {name='self', type='tnt.SplitDataset'},
   call =
      function(self)
         return self.__partitionsizes[self.__partition]
      end
}

SplitDataset.get = argcheck{
   {name='self', type='tnt.SplitDataset'},
   {name='idx', type='number'},
   call =
      function(self, idx)
         assert(idx >= 1 and idx <= self:size(), 'index out of bounds')
         local offset = (self.__partition == 1) and 0 or
            self.__partitionsizes:narrow(1, 1, self.__partition - 1):sum()
         return self.__dataset:get(offset + idx)
      end
}
