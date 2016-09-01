--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local AverageValueMeter = torch.class('tnt.AverageValueMeter', 'tnt.Meter', tnt)

AverageValueMeter.__init = argcheck{
   doc = [[
<a name="AverageValueMeter">
#### tnt.AverageValueMeter(@ARGP)
@ARGT

The `tnt.AverageValueMeter` measures and returns the average value and the
standard deviation of any collection of numbers that are `add`ed to it. It is
useful, for instance, to measure the average loss over a collection of examples.

The `add()` function expects as input a Lua number `value`, which is the value
that needs to be added to the list of values to average. It also takes as input
an optional parameter `n` that assigns a weight to `value` in the average, in
order to facilitate computing weighted averages (default = 1).

The `tnt.AverageValueMeter` has no parameters to be set at initialization time.
]],
   {name="self", type="tnt.AverageValueMeter"},
   call =
      function(self)
         self:reset()
      end
}

AverageValueMeter.reset = argcheck{
   {name="self", type="tnt.AverageValueMeter"},
   call =
      function(self)
         self.sum = 0
         self.n = 0
         self.var = 0
      end
}

AverageValueMeter.add = argcheck{
   {name="self", type="tnt.AverageValueMeter"},
   {name="value", type="number"},
   {name="n", type="number", default=1},
   call =
      function(self, value, n)
         self.sum = self.sum + value
         self.var = self.var + value * value
         self.n = self.n + n
      end
}

AverageValueMeter.value = argcheck{
    {name="self", type="tnt.AverageValueMeter"},
    call =
    function(self)
        local n = self.n
        local mean = self.sum / n
        -- unbiased estimator of the variance:
        local std = math.sqrt( (self.var - n * mean * mean) / (n-1) )
        return mean, std
    end
}
