--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local PrecisionAtKMeter = torch.class('tnt.PrecisionAtKMeter', 'tnt.Meter', tnt)

PrecisionAtKMeter.__init = argcheck{
   doc = [[
<a name="PrecisionAtKMeter">
#### tnt.PrecisionAtKMeter(@ARGP)
@ARGT

The `tnt.PrecisionAtKMeter` measures the precision@k of ranking methods at pre-specified
levels k. The precision@k is the percentage of the k front-ranked
items according to the model that is in the list of correct (positive) targets.

At initialization time, a table `topk` may be given as input that specifies the
levels k at which the precision@k will be measures (default = `{10}`). In
addition, a number `dim` may be provided that specifies over which dimension the
precision@k should be computed (default = 2), and a boolean `online` may be
specified that indicates whether we see all inputs along dimension `dim` at once
(default = `false`).

The `add(output, target)` method takes two inputs. In the default mode (`dim=2`
and `online=false`), the inputs mean:
   * A NxC tensor that for each of the N examples (queries) contains a score
     indicating to what extent each of the C classes (documents) is relevant to
     the query, according to the model.
   * A binary NxC `target` tensor that encodes which of the C classes
     (documents) are actually relevant to the the N-th input (query). For
     instance, a row of {0, 1, 0, 1} indicates that the example is associated
     with classes 2 and 4.

The result of setting `dim` to `1` is identical to transposing the tensors
`output` and `target` in the above. The result of setting `online=true` is that
the function assumes that it is not the number of queries `N` that is growing
with repeated calls to `add()`, but the number of candidate documents `C`. (Use
this mode in scenarios where `C` is large but `N` is small.)

The `value()` method returns a table that contains the precision@k (that is, the
percentage of targets predicted correctly) at the cutoff levels in `topk` that
were specified at initialization time. Alternatively, the precision@k at
a specific level k can be obtained by calling `value(k)`. Note that the level
`k` should be an element of the table `topk` specified at initialization time.

Please note that the maximum value in `topk` cannot be higher than the total
number of classes (documents).
]],
   noordered = true,
   {name="self",   type="tnt.PrecisionAtKMeter"},
   {name="topk",   type="table",   default={10}},
   {name="dim",    type="number",  default=2},
   {name="online", type="boolean", default=false},
   call =
      function(self, topk, dim, online)
         assert(dim == 1 or dim == 2, 'value of dimension should be 1 or 2')
         self.topk   = torch.LongTensor(topk):sort():totable()
         self.maxk   = self.topk[#self.topk]
         self.dim    = dim
         self.online = online
         self:reset()
      end
}

PrecisionAtKMeter.reset = argcheck{
   {name="self", type="tnt.PrecisionAtKMeter"},
   call =
      function(self)
         self.tp = {}
         for _,k in ipairs(self.topk) do self.tp[k] = 0 end
         self.n = 0
      end
}

PrecisionAtKMeter.add = argcheck{
   {name="self",   type="tnt.PrecisionAtKMeter"},
   {name="output", type="torch.*Tensor"},
   {name="target", type="torch.*Tensor"},  -- target is k-hot vector
   call =
      function(self, output, target)
         output = output:squeeze()
         if output:nDimension() == 1 then
             output = output:view(1, output:size(1))
         else
             assert(
                 output:nDimension() == 2,
                 'wrong output size (1D or 2D expected)'
             )
         end
         if target:nDimension() == 1 then
             target = target:view(1, target:size(1))
         else
             assert(
                 target:nDimension() == 2,
                 'wrong target size (1D or 2D expected)'
             )
         end
         for s = 1,#output:size() do
             assert(output:size(s) == target:size(s),
                 string.format('target and output do not match on dim %d', s))
         end

         -- add new output-target pairs:
         if self.online then

            -- update top-k list along dimension dim:
            self:__updatetopk{
               output = output,
               target = target,
               topk   = self.maxk,
               dim    = self.dim,
               desc   = true,
            }
            self.n = output:size((self.dim == 1) and 2 or 1)
         else

            -- accumulate counts of true positives and total # of inputs:
            local topout, topind = torch.topk(output, self.maxk, self.dim, true)
            local _,sortind = torch.sort(topout, self.dim, true)
            local topind = topind:gather(self.dim, sortind)
            local sorttarget = target:gather(self.dim, topind)
            for _,k in ipairs(self.topk) do
               self.tp[k] = self.tp[k] + sorttarget:narrow(self.dim, 1, k):sum()
            end
            self.n = self.n + target:size((self.dim == 1) and 2 or 1)
         end
      end
}

PrecisionAtKMeter.value = argcheck{
   {name="self", type="tnt.PrecisionAtKMeter"},
   {name="k", type="number", opt=true},
   call =
      function(self, k)

         -- in online mode, sort outputs and corresponding targets:
         if self.online then
            local topoutput = self.__topkoutput:narrow(self.dim, 1, self.maxk)
            local toptarget = self.__topktarget:narrow(self.dim, 1, self.maxk)
            local _,sortind = torch.sort(topoutput, self.dim, true)
            local sorttarget = toptarget:gather(self.dim, sortind)
            for _,k in ipairs(self.topk) do
               self.tp[k] = sorttarget:narrow(self.dim, 1, k):sum()
            end
         end

         -- compute the precision@k:
         if k then
            assert(self.tp[k], 'invalid k (not provided at construction time)')
            return (self.tp[k] / (self.n * k)) * 100
         else
            local value = {}
            for _,k in ipairs(self.topk) do value[k] = self:value(k) end
            return value
         end
      end
}
