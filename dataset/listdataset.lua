--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local tds = require 'tds'
local argcheck = require 'argcheck'
local transform = require 'torchnet.transform'

local ListDataset, Dataset = torch.class('tnt.ListDataset', 'tnt.Dataset', tnt)

ListDataset.__init = argcheck{
   doc = [[
<a name="ListDataset">
#### tnt.ListDataset(@ARGP)
@ARGT

Considering a `list` (can be a `tds.Hash`, `table` or a `torch.LongTensor`) the
i-th sample of a dataset will be returned by `load(list[i])`, where `load()` is
a closure provided by the user.

If `path` is provided, list is assumed to be a list of string, and will
each element `list[i]` will prefixed by `path/` when fed to `load()`.

Purpose: many low or medium-scale datasets can be seen as a list of files
(for example representing input samples). For this list of file, a target
can be often inferred in a simple manner.

]],
   {name='self', type='tnt.ListDataset'},
   {name='list', type='tds.Hash'},
   {name='load', type='function'},
   {name='path', type='string', opt=true},
   call =
      function(self, list, load, path)
         Dataset.__init(self)
         self.list = list
         self.load = load
         self.path = path
      end
}

ListDataset.__init = argcheck{
   {name='self', type='tnt.ListDataset'},
   {name='list', type='table'},
   {name='load', type='function'},
   {name='path', type='string', opt=true},
   overload = ListDataset.__init,
   call =
      function(self, list, load, path)
         Dataset.__init(self)
         self.list = list
         self.load = load
         self.path = path
      end
}

ListDataset.__init = argcheck{
   {name='self', type='tnt.ListDataset'},
   {name='list', type='torch.LongTensor'},
   {name='load', type='function'},
   {name='path', type='string', opt=true},
   overload = ListDataset.__init,
   call =
      function(self, list, load, path)
         Dataset.__init(self)
         self.list = list
         self.load = load
         self.path = path
      end
}

ListDataset.__init = argcheck{
   doc = [[
#### tnt.ListDataset(@ARGP)
@ARGT

The file specified by `filename` is interpreted as a list of strings (one
string per line). The i-th sample of a dataset will be returned by
`load(line[i])`, where `load()` is a closure provided by the user an
`line[i]` is the i-the line of `filename`.

If `path` is provided, list is assumed to be a list of string, and will
each element `list[i]` will prefixed by `path/` when fed to `load()`.

]],
   {name='self', type='tnt.ListDataset'},
   {name='filename', type='string'},
   {name='load', type='function'},
   {name='maxload', type='number', opt=true},
   {name='path', type='string', opt=true},
   overload = ListDataset.__init,
   call =
      function(self, filename, load, maxload, path)
         local list = tds.hash()
         for filename in io.lines(filename) do
            list[#list+1] = filename
            if maxload and maxload > 0 and #list == maxload then
               break
            end
         end
         ListDataset.__init(self, list, load, path)
         print(string.format("| loaded <%s> with %d examples", filename, #list))
      end
}

ListDataset.size = argcheck{
   {name='self', type='tnt.ListDataset'},
   call =
      function(self)
         return torch.isTensor(self.list) and self.list:size(1)
                                           or #self.list
      end
}

ListDataset.get = argcheck{
   {name='self', type='tnt.ListDataset'},
   {name='idx', type='number'},
   call =
      function(self, idx)
         assert(idx >= 1 and idx <= self:size(), 'out of bound')
         if self.path then
            return self.load(string.format("%s/%s", self.path, self.list[idx]))
         else
            return self.load(self.list[idx])
         end
      end
}
