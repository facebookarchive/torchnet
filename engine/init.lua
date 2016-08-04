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

doc[[

### tnt.Engine

In experimenting with different models and datasets, the underlying training
procedure is often the same. The `Engine` module provides the boilerplate logic
necessary for the training and testing of models. This might include conducting
the interaction between model (`nn.Module`), `tnt.DatasetIterator`s,
`nn.Criterion`s, and `tnt.Meter`s.

An instance `engine` of a `tnt.Engine()` implements two main methods:

  * `engine:train()`, for training the model on data
        (i.e. sample data, forward prop, backward prop).
  * `engine:test()`,  for evaluating a model on data
        (optionally with respect to a `nn.Criterion`).

The `Engine` can be implemented for any common underlying training and testing
procedure involving a model and data. It can also be designed to allow user
control after certain events such as forward prop, criterion evaluation, or the
end of an epoch, by using coroutines (see `tnt.SGDEngine`).

]]

local Engine = torch.class('tnt.Engine', tnt)

Engine.__init = argcheck{
   nonamed=true, -- to avoid ambiguities
   {name="self", type="tnt.Engine"},
   {name="hooks", type="table"},
   call =
      function(self, hooks)
         self.hooks = {}
         for _, name in ipairs(hooks) do
            assert(type(name) == 'string', 'hooks must be a table of hook names (strings)')
            self.hooks[name] = function() end
         end
         setmetatable(
            self.hooks,
            {
               __index =
                  function(hooks, name)
                     assert(type(name) == 'string', 'hook name must be a string')
                     error(string.format('unknown hook <%s>', name))
                  end,
               __newindex =
                  function(self, name)
                     assert(type(name) == 'string', 'hook name must be a string')
                     error(string.format('unknown hook <%s>', name))
                  end,
               __call =
                  function(hooks, name, ...)
                     return hooks[name](...)
                  end
            }
         )
      end
}

Engine.train = argcheck{
   {name="self", type="tnt.Engine"},
   call =
      function(self)
         error('A tnt.Engine should implement the train() function.')
      end
}

Engine.test = argcheck{
   {name="self", type="tnt.Engine"},
   call =
      function(self)
         error('A tnt.Engine should implement the test() function.')
      end
}
