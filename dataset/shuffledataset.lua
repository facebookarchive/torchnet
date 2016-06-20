--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local ShuffleDataset, ResampleDataset =
   torch.class('tnt.ShuffleDataset', 'tnt.ResampleDataset', tnt)

ShuffleDataset.__init = argcheck{
   doc = [[
<a name="ShuffleDataset">
#### tnt.ShuffleDataset(@ARGP)
@ARGT

`tnt.ShuffleDataset` is a sub-class of
[tnt.ResampleDataset](#ResampleDataset) provided for convenience.

It samples uniformly from the given `dataset` with, or without
`replacement`. The chosen partition can be redrawn by calling
[resample()](#ShuffleDataset.resample).

If `replacement` is `true`, then the specified `size` may be larger than
the underlying `dataset`.

If `size` is not provided, then the new dataset size will be equal to the
underlying `dataset` size.

Purpose: the easiest way to shuffle a dataset!
]],
   {name='self', type='tnt.ShuffleDataset'},
   {name='dataset', type='tnt.Dataset'},
   {name='size', type='number', opt=true},
   {name='replacement', type='boolean', default=false},
   call =
      function(self, dataset, size, replacement)
         if size and not replacement and size > dataset:size() then
            error('size cannot be larger than underlying dataset size when sampling without replacement')
         end
         self.__replacement = replacement
         local function sampler(dataset, idx)
            return self.__perm[idx]
         end
         ResampleDataset.__init(self, {
            dataset = dataset,
            sampler = sampler,
            size    = size})
         self:resample()
      end
}

ShuffleDataset.resample = argcheck{
   doc = [[
<a name="ShuffleDataset.resample">
##### tnt.ShuffleDataset.resample(@ARGP)

The permutation associated to `tnt.ShuffleDataset` is fixed, such that two
calls to the same index will return the same sample from the underlying
dataset.

Call `resample()` to draw randomly a new permutation.
]],
   {name='self', type='tnt.ShuffleDataset'},
   call =
      function(self)
         self.__perm = self.__replacement
            and torch.LongTensor(self:size()):random(self.__dataset:size())
            or  torch.randperm(self.__dataset:size()):long():narrow(1, 1, self:size())
      end
}
