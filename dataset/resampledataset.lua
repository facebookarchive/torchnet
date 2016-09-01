--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local ResampleDataset =
   torch.class('tnt.ResampleDataset', 'tnt.Dataset', tnt)

ResampleDataset.__init = argcheck{
   doc = [[
<a name="ResampleDataset">
#### tnt.ResampleDataset(@ARGP)

Given a `dataset`, creates a new dataset which will (re-)sample from this
underlying dataset using the provided `sampler(dataset, idx)` closure.

If `size` is provided, then the newly created dataset will have the
specified `size`, which might be different than the underlying dataset
size.

If `size` is not provided, then the new dataset will have the same size
than the underlying one.

By default `sampler(dataset, idx)` is the identity, simply `return`ing `idx`.
`dataset` corresponds to the underlying dataset provided at construction, and
`idx` may take a value between 1 to `size`. It must return an index in the range
acceptable for the underlying dataset.

Purpose: shuffling data, re-weighting samples, getting a subset of the
data. Note that an important sub-class is ([tnt.ShuffleDataset](#ShuffleDataset)),
provided for convenience.
]],
   {name='self', type='tnt.ResampleDataset'},
   {name='dataset', type='tnt.Dataset'},
   {name='sampler', type='function', default=function(dataset, idx) return idx end},
   {name='size', type='number', opt=true},
   call =
      function(self, dataset, sampler, size)
         self.__sampler = sampler
         self.__dataset = dataset
         self.__size = size
      end
}

ResampleDataset.size = argcheck{
   {name='self', type='tnt.ResampleDataset'},
   call =
      function(self)
         return (self.__size and self.__size > 0) and self.__size or self.__dataset:size()
      end
}

ResampleDataset.get = argcheck{
   {name='self', type='tnt.ResampleDataset'},
   {name='idx', type='number'},
   call =
      function(self, idx)
         assert(idx >= 1 and idx <= self:size(), 'index out of bound')
         idx = self.__sampler(self.__dataset, idx)
         assert(idx >= 1 and idx <= self.__dataset:size(), 'index out of bound (sampler)')
         return self.__dataset:get(idx)
      end
}
