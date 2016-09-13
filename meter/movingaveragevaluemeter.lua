--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local MovingAverageValueMeter = torch.class(
                                  'tnt.MovingAverageValueMeter',
                                  'tnt.Meter', tnt
                                )

MovingAverageValueMeter.__init = argcheck{
   doc = [[
<a name="MovingAverageValueMeter">
#### tnt.MovingAverageValueMeter(@ARGP)
@ARGT

The `tnt.MovingAverageValueMeter` measures and returns the average value
and the standard deviation of any collection of numbers that are `add`ed to it
within the most recent moving average window. It is useful, for instance,
to measure the average loss over a collection of examples withing the
most recent window.

The `add()` function expects as input a Lua number `value`, which is the value
that needs to be added to the list of values to average.

The `tnt.MovingAverageValueMeter` needs the moving window size to be set at
initialization time.
]],
   {name="self", type="tnt.MovingAverageValueMeter"},
   {name="windowsize", type="number"},
   call =
      function(self, windowsize)
         self.windowsize = windowsize;
         self.valuequeue = torch.Tensor(self.windowsize)
         self:reset()
      end
}

MovingAverageValueMeter.reset = argcheck{
   {name="self", type="tnt.MovingAverageValueMeter"},
   call =
      function(self)
         self.sum = 0
         self.n = 0
         self.var = 0
         self.valuequeue:fill(0.)
      end
}

MovingAverageValueMeter.add = argcheck{
   {name="self", type="tnt.MovingAverageValueMeter"},
   {name="value", type="number"},
   call =
      function(self, value)
         local queueid = (self.n % self.windowsize) + 1
         local oldvalue = self.valuequeue[queueid]
         self.sum = self.sum + value - oldvalue
         self.var = self.var + value * value
                    - oldvalue * oldvalue
         self.valuequeue[queueid] = value
         self.n = self.n + 1
      end
}

MovingAverageValueMeter.value = argcheck{
    {name="self", type="tnt.MovingAverageValueMeter"},
    call =
       function(self)
           local n = math.min(self.n, self.windowsize)
           local mean = self.sum / math.max(1, n)
           -- unbiased estimator of the variance:
           local std = math.sqrt((self.var - n * mean * mean) / math.max(1, n-1))
           return mean, std
       end
}
