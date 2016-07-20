--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'
local utils = require 'torchnet.utils'

local TransformDataset =
   torch.class('tnt.TransformDataset', 'tnt.Dataset', tnt)

TransformDataset.__init = argcheck{
   doc = [[
<a name="TransformDataset">
#### tnt.TransformDataset(@ARGP)
@ARGT

Given a closure `transform()`, and a `dataset`, `tnt.TransformDataset`
applies the closure in an on-the-fly manner when querying a sample with
`tnt.Dataset:get()`.

If key is provided, the closure is applied to the sample field specified
by `key` (only). The closure must return the new corresponding field value.

If key is not provided, the closure is applied on the full sample. The
closure must return the new sample table.

The size of the new dataset is equal to the size of the underlying `dataset`.

Purpose: when performing pre-processing operations, it is convenient to be
able to perform on-the-fly transformations to a
dataset.
]],
   {name='self', type='tnt.TransformDataset'},
   {name='dataset', type='tnt.Dataset'},
   {name='transform', type='function'},
   {name='key', type='string', opt=true},
   call =
      function(self, dataset, transform, key)
         self.dataset = dataset
         if key then
            function self.__transform(z, idx)
               assert(z[key], 'missing key in sample')
               z[key] = transform(z[key], idx)
               return z
            end
         else
            function self.__transform(z, idx)
               return transform(z, idx)
            end
         end
      end
}

TransformDataset.__init = argcheck{
   doc = [[
<a name="TransformDataset">
#### tnt.TransformDataset(@ARGP)
@ARGT

Given a set of closures and a `dataset`, `tnt.TransformDataset` applies
these closures in an on-the-fly manner when querying a sample with
`tnt.Dataset:get()`.

Closures are provided in `transforms`, a Lua table, where a (key,value)
pair represents a (sample field name, corresponding closure to be applied
to the field name).

Each closure must return the new value of the corresponding field.
]],
   {name='self', type='tnt.TransformDataset'},
   {name='dataset', type='tnt.Dataset'},
   {name='transforms', type='table'},
   overload = TransformDataset.__init,
   call =
      function(self, dataset, transforms)
         for k,v in pairs(transforms) do
            assert(type(v) == 'function',
                   'key/function table expected for transforms')
         end
         self.dataset = dataset
         transforms = utils.table.copy(transforms)
         function self.__transform(z)
            for key,transform in pairs(transforms) do
               assert(z[key], 'missing key in sample')
               z[key] = transform(z[key])
            end
            return z
         end
      end
}

TransformDataset.size = argcheck{
   {name='self', type='tnt.TransformDataset'},
   call =
      function(self)
         return self.dataset:size()
      end
}

TransformDataset.get = argcheck{
   {name='self', type='tnt.TransformDataset'},
   {name='idx', type='number'},
   call =
      function(self, idx)
         return self.__transform(
            self.dataset:get(idx), idx
         )
      end
}
