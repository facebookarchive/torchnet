--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local argcheck = require 'argcheck'
local doc = require 'argcheck.doc'

local utable = {}

doc[[
<a name="utils.table.clone">
#### tnt.utils.table.clone(table)

This function do a deep copy of a table.

]]

function utable.clone(tbl)
    return torch.deserializeFromStorage(torch.serializeToStorage(tbl))
end

function utable.copy(tbl)
    local cpy = {}
    for k,v in pairs(tbl) do
        cpy[k] = v
    end
    return cpy
end

utable.merge = argcheck{
    doc = [[
<a name="utils.table.merge">
#### tnt.utils.table.merge(@ARGP)
@ARGT

This function add to the destination table `dest`, the
element contained in the source table `source`.

The copy is shallow.

If a key exists in both tables, then the element in the source table
is preferred.
]],
    {name = "dst", type = 'table'},
    {name = "src", type = 'table'},
    call = function (dst, src)
        for k,v in pairs(src) do
            dst[k] = v
        end
        return dst
    end
}

utable.foreach = argcheck{
    doc = [[
<a name="utils.table.foreach">
#### tnt.utils.table.foreach(@ARGP)
@ARGT

This function applies the function defined by `closure` to the
table `tbl`.

If `recursive` is given and set to `true`, the `closure` function
will be apply recursively to the table.
]],
    {name = "tbl",       type = 'table'},
    {name = "closure",   type = 'function'},
    {name = "recursive", type = 'boolean', default = false},
    call = function(tbl, closure, recursive)
        local newtbl = {}
        for k,v in pairs(tbl) do
            if recursive and type(v) == 'table' then
                newtbl[k] = utable.foreach(v, closure, recursive)
            else
                newtbl[k] = closure(v)
            end
        end
        return newtbl
    end
}

doc[[
<a name="utils.table.canmergetensor">
#### tnt.utils.table.canmergetensor(tbl)

Check if a table can be merged into a tensor.
]]

function utable.canmergetensor(tbl)
   if type(tbl) ~= 'table' then
      return false
   end

   local typename = torch.typename(tbl[1])
   if typename and typename:match('Tensor') then
      local sz = tbl[1]:nElement()
      for i=2,#tbl do
         -- can't merge tensors of different sizes
         if tbl[i]:nElement() ~= sz then
            return false
         end
      end
      return true
   end
   return false
end

utable.mergetensor = argcheck{
    doc = [[
<a name="utils.table.mergetensor">
#### tnt.utils.table.mergetensor(@ARGP)
@ARGT

Merge a table into a tensor in one extra dimension.
]],
    {name = 'tbl', type = 'table'},
    call = function(tbl)
        local sz = tbl[1]:size():totable()
        table.insert(sz, 1, #tbl)
        sz = torch.LongStorage(sz)
        local res = tbl[1].new():resize(sz)
        for i=1,#tbl do
            res:select(1, i):copy(tbl[i])
        end
        return res
    end
}

return utable
