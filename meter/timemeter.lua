--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local TimeMeter = torch.class('tnt.TimeMeter', 'tnt.Meter', tnt)

TimeMeter.__init = argcheck{
   doc = [[
<a name="TimeMeter">
#### tnt.TimeMeter(@ARGP)
@ARGT

The `tnt.TimeMeter` is designed to measure the time between events and can be
used to measure, for instance, the average processing time per batch of data.
It is different from most other meters in terms of the methods it provides:

At initialization time, an optional boolean parameter `unit` may be provided
(default = `false`). When set to `true`, the value returned by the meter
will be divided by the number of times that the `incUnit()` method is called.
This allows the user to compute, for instance, the average processing time per
batch by simply calling the `incUnit()` method after processing a batch.

The `tnt.TimeMeter` provides the following methods:

   * `reset()` resets the timer, setting the timer and unit counter to zero.
   * `stop()` stops the timer.
   * `resume()` resumes the timer.
   * `incUnit()` increments the unit counter by one.
   * `value()` returns the time passed since the last `reset()`; divided by the counter value when `unit=true`.
]],
   {name="self", type="tnt.TimeMeter"},
   {name="unit", type="boolean", default=false},
   call =
      function(self, unit)
         self.unit = unit
         self.timer = torch.Timer()
         self:reset()
      end
}

TimeMeter.reset = argcheck{
   {name="self", type="tnt.TimeMeter"},
   call =
      function(self)
         self.timer:reset()
         self.n = 0
      end
}

TimeMeter.stop = argcheck{
   {name="self", type="tnt.TimeMeter"},
   call =
      function(self)
         self.timer:stop()
      end
}

TimeMeter.resume = argcheck{
   {name="self", type="tnt.TimeMeter"},
   call =
      function(self)
         self.timer:resume()
      end
}

TimeMeter.incUnit = argcheck{
   {name="self", type="tnt.TimeMeter"},
   {name="value", type="number", default=1},
   call =
      function(self, value)
         self.n = self.n + value
      end
}

TimeMeter.value = argcheck{
   {name="self", type="tnt.TimeMeter"},
   call =
      function(self)
         local time = self.timer:time().real
         if self.unit then
            return time/self.n
         else
            return time
         end
      end
}
