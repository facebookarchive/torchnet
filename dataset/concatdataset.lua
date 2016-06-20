--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local ConcatDataset =
   torch.class('tnt.ConcatDataset', 'tnt.Dataset', tnt)

ConcatDataset.__init = argcheck{
   doc = [[
<a name="ConcatDataset">
#### tnt.ConcatDataset(@ARGP)
@ARGT

Given a Lua array (`datasets`) of [tnt.Dataset](#Dataset), concatenates
them into a single dataset.  The size of the new dataset is the sum of the
underlying dataset sizes.

Purpose: useful to assemble different existing datasets, possibly
large-scale datasets as the concatenation operation is done in an
on-the-fly manner.
]],
   noordered=true,
   {name='self', type='tnt.ConcatDataset'},
   {name='datasets', type='table'},
   call =
      function(self, datasets)
         assert(#datasets > 0, 'datasets should not be an empty table')
         local indices = torch.LongTensor(#datasets, 2) -- indices: begin, end
         local size = 0
         for i, dataset in ipairs(datasets) do
            assert(torch.isTypeOf(dataset, 'tnt.Dataset'),
                   'each member of datasets table should be a tnt.Dataset')
            indices[i][1] = size+1
            size = size + dataset:size()
            indices[i][2] = size
         end
         self.__datasets = datasets
         self.__indices = indices
         self.__size = size
      end
}

ConcatDataset.size = argcheck{
   {name='self', type='tnt.ConcatDataset'},
   call =
      function(self)
         return self.__size
      end
}

ConcatDataset.get = argcheck{
   {name='self', type='tnt.ConcatDataset'},
   {name='idx', type='number'},
   call =
      function(self, idx)
         assert(idx >= 1 and idx <= self.__size, 'index out of bound')
         local indices = self.__indices
         local l, r = 1, indices:size(1)
         while l ~= r do
            local m = math.floor((r-l)/2) + l
            if l == m then
               if idx > indices[l][2] then
                  l, r = r, r
               else
                  l, r = l, l
               end
            else
               if idx > indices[m][2] then
                  l, r = m, r
               elseif idx < indices[m][1] then
                  l, r = l, m
               else
                  l, r = m, m
               end
            end
         end
         return self.__datasets[l]:get(idx-self.__indices[l][1]+1)
      end
}
