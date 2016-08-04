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
local doc = require 'argcheck.doc'

doc[[

### tnt.transform

*Torchnet* provides a set of general data transformations.
These transformations are either directly on the data (e.g., normalization)
or on their structure. This is particularly handy
when manipulating [tnt.Dataset](#tnt.Dataset).

Most of the transformations are simple but can be [composed](#transform.compose) or
[merged](#transform.merge).
]]

local transform = {}
tnt.transform = transform
local unpack = unpack or table.unpack


doc[[
<a name="transform.identity">
#### transform.identity(...)

The identity transform takes any input and return it as it is.

For example, this function is useful when composing
transformations on data from multiple sources, and some of the sources
must not be transformed.
]]

transform.identity =
   function(...)
      local args = {...}
      return function()
         return unpack(args)
      end
   end

transform.compose = argcheck{
   doc = [[
<a name = "transform.compose">
#### transform.compose(@ARGP)
@ARGT

This function takes a `table` of functions and
composes them to return one transformation.

This function assumes that the table of transformations
is indexed by contiguous ordered keys starting at 1.
The transformations are composed in the ascending order.

For example, the following code:
```lua
> f = transform.compose{
        [1] = function(x) return 2*x end,
        [2] = function(x) return x + 10 end,
        foo = function(x) return x / 2 end,
        [4] = function(x) return x - x end
   }
   > f(3)
   16
```
is equivalent to compose the transformations stored in [1] and [2], i.e.,
defining the following transformation:
```lua
> f =  function(x) return 2*x + 10 end
```
Note that transformations stored with keys `foo` and `4` are ignored.
]],
   {name='transforms', type='table'},
   call =
      function(transforms)
         for k,v in ipairs(transforms) do
            assert(type(v) == 'function', 'table of functions expected')
         end
         transforms = utils.table.copy(transforms)
         return
            function(z)
               for _, trans in ipairs(transforms) do
                  z = trans(z)
               end
               return z
            end
      end
}

transform.merge = argcheck{
   doc = [[
<a name = "transform.merge">
#### transform.merge(@ARGP)
@ARGT

This function takes a `table` of transformations and
merge them into one transformation.
Once apply to an input, this transformation will produce a `table` of output,
containing the transformed input.

For example, the following code:
```lua
> f = transform.merge{
        [1] = function(x) return 2*x end,
        [2] = function(x) return x + 10 end,
        foo = function(x) return x / 2 end,
        [4] = function(x) return x - x end
   }
```
produces a function which applies a set of transformations to the same input:
```lua
> f(3)
   {
     1 : 6
     2 : 13
     foo : 1.5
     4 : 0
   }
```
]],
   {name='transforms', type='table'},
   call =
      function(transforms)
         for k,v in pairs(transforms) do
            assert(type(v) == 'function', 'table of functions expected')
         end
         transforms = utils.table.copy(transforms)
         return
            function(z)
               local newz = {}
               for k, trans in pairs(transforms) do
                  newz[k] = trans(z)
               end
               return utils.table.mergetensor(newz)
            end
      end
}

transform.tablenew = argcheck{
   doc = [[
<a name = "transform.tablenew">
#### transform.tablenew()

This function creates a new table of functions from an
existing table of functions.
]],
   call =
      function()
         return
            function(func)
               local tbl = {}
               for k,v in pairs(func) do
                  tbl[k] = v
               end
               return tbl
            end
      end
}

transform.tableapply = argcheck{
   doc = [[
<a name = "transform.tableapply">
#### transform.tableapply(@ARGP)
@ARGT

This function applies a transformation to a table of input.
It return a table of output of the same size as the input.

For example, the following code:
```lua
> f = transform.tableapply(function(x) return 2*x end)
```
produces a function which multiplies any input by 2:
```lua
> f({[1] = 1, [2] = 2, foo = 3, [4] = 4})
   {
     1 : 2
     2 : 4
     foo : 6
     4 : 8
   }
```
]],
   {name='transform', type='function'},
   call =
      function(transform)
         return
            function(tbl)
               return utils.table.foreach(tbl, transform)
            end
      end
}

transform.tablemergekeys = argcheck{
   doc = [[
<a name = "transform.tablemergekeys">
#### transform.tablemergekeys()

This function merges tables by key. More precisely, the input must be a
`table` of `table` and this function will reverse the table orderto
make the keys from the nested table accessible first.

For example, if the input is:
```lua
> x = { sample1 = {input = 1, target = "a"} , sample2 = {input = 2, target = "b", flag = "hard"}
```
Then apply this function will produce:
```lua
> transform.tablemergekeys(x)
{
   input :
         {
           sample1 : 1
           sample2 : 2
         }
   target :
          {
            sample1 : "a"
            sample2 : "b"
          }
   flag :
        {
           sample2: "hard"
        }
}
```
]],
   call =
      function()
         return
            function(tbl)
               local mergedtbl = {}
               for idx, elem in ipairs(tbl) do
                  for key, value in pairs(elem) do
                     if not mergedtbl[key] then
                        mergedtbl[key] = {}
                     end
                     mergedtbl[key][idx] = value
                  end
               end
               return mergedtbl
            end
      end
}

transform.makebatch = argcheck{
   doc = [[
<a name = "transform.makebatch">
#### transform.makebatch(@ARGP)
@ARGT

This function is used in many `tnt.Dataset` to format
samples in the format used by the `tnt.Engine`.

This function first [merges keys](#transform.tablemergekeys) to
produces a table of output. Then, transform this table into a tensor by
either using a `merge` transformation provided by the user or by
simply concatenating the table into a tensor directly.

This function uses the [compose](#transform.compose) transform to apply
successive transformations.
]],
   {name='merge', type='function', opt=true},
   call =
      function(merge)

         local makebatch
         if merge then
            makebatch = transform.compose{
               transform.tablemergekeys(),
               merge
            }
         else
            makebatch = transform.compose{
               transform.tablemergekeys(),
               transform.tableapply(
                  function(field)
                     if utils.table.canmergetensor(field) then
                        return utils.table.mergetensor(field)
                     else
                        return field
                     end
                  end
               )
            }
         end

         return
            function(samples)
               assert(type(samples) == 'table', 'makebatch: table of samples expected')
               return makebatch(samples)
            end
      end
}

transform.randperm = argcheck{
   doc = [[
<a name = "transform.perm">
#### transform.perm(@ARGP)
@ARGT

This function create a vector containing a permutation of the indices from 1 to `size`.
This vector is a `LongTensor` and  `size` must be a number.

Once the vector created, this function can be used to call a specific indices in it.

For example:
```lua
> p = transform.perm(3)
```
creates a function `p` which contains a permutation of indices:
```lua
> p(1)
2
> p(2)
1
> p(3)
3
```
]],
   {name="size", type="number"},
   call =
      function(size)
         local perm = torch.randperm(size, 'torch.LongTensor')
         return
            function(idx)
               return perm[idx]
            end
      end
}

transform.normalize = argcheck{
   doc = [[
<a name = "transform.normalize">
#### transform.normalize(@ARGP)
@ARGT

This function normalizes data, i.e., it removes its mean and
divide it by its standard deviation.

The input must be a `Tensor`.

Once create, a `threshold` can be given (must be a number). Then,
the data will be divided by their standard deviation, only if this
deviation is greater than the `threshold`. This is handy, if the
deviation is small and deviding by it could lead to unstability.
]],
   {name='threshold', type='number', default=0},
   call =
      function(threshold)
         return
            function(z)
               local std = z:std()
               z:add(-z:mean())
               if std > threshold then
                  z:div(std)
               end
               return z
            end
      end
}

return transform
