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

local TableDataset, ListDataset
    = torch.class('tnt.TableDataset', 'tnt.ListDataset', tnt)

TableDataset.__init = argcheck{
    doc = [[
<a name = "TableDataset">
#### tnt.TableDataset(@ARGP)
@ARGT

`tnt.TableDataset` interfaces existing data
to torchnet. It is useful if you want to use torchnet on a small dataset.

The data must be contained in a `tds.Hash`.

`tnt.TableDataset` does a shallow copy of the data.

Data are loaded while constructing the `tnt.TableDataset`:
```lua
> a = tnt.TableDataset{data = {1,2,3}}
> print(a:size())
3
```
`tnt.TableDataset` assumes that table has contiguous keys starting at 1.
]],
    noordered = true,
    {name = 'self', type = 'tnt.TableDataset'},
    {name = 'data', type = 'table'},
    call = function(self, data)
        for i = 1, #data do
            assert(data[i], "keys are not contiguous integers starting at 1")
        end
        local size = 0
        for _, _ in pairs(data) do size = size + 1 end
        assert(size == #data, "keys are not contiguous integers starting at 1")
        self.data = data
    end
}

TableDataset.size = argcheck{
    {name = 'self', type = 'tnt.TableDataset'},
    call = function(self)
        return #self.data
    end
}

TableDataset.get = argcheck{
    {name = 'self', type = 'tnt.TableDataset'},
    {name = 'idx',  type = 'number'},
    call = function(self, idx)
        assert(idx >= 1 and idx <= self:size(), 'index out of bound')
        assert(idx == math.floor(idx), 'index must be an integer')
        return utils.table.clone(self.data[idx])
    end
}
