--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local ClassErrorMeter = torch.class('tnt.ClassErrorMeter', 'tnt.Meter', tnt)

ClassErrorMeter.__init = argcheck{
   doc = [[
<a name="ClassErrorMeter">
#### tnt.ClassErrorMeter(@ARGP)
@ARGT

The `tnt.ClassErrorMeter` measures the classification error (in %) of
classification models (zero-one loss). The meter can also measure the error of
predicting the correct label among the top-k scoring labels (for instance, in
the Imagenet competition, one generally measures classification@5 errors).

At initialization time, it takes to optional parameters: (1) a table
`topk` that contains the values at which the classification@k errors should be
measures (default = {1}); and (2) a boolean `accuracy` that makes the meter
output accuracies instead of errors (accuracy = 1 - error).

The `add(output, target)` method takes as input an NxK-tensor `output` that
contains the output scores for each of the N examples and each of the K classes,
and an N-tensor `target` that contains the targets corresponding to each of the
N examples (targets are integers between 1 and K). If only one example is
`add`ed, `output` may also be a K-tensor and target a 1-tensor.

Please note that `topk` (if specified) may not contain values larger than K.

The `value()` returns a table with the classification@k errors for all values
at k that were specified in `topk` at initialization time. Alternatively,
`value(k)` returns the classification@k error as a number; only values of `k`
that were element of `topk` are allowed. If `accuracy` was set to `true` at
initialization time, the `value()` method returns accuracies instead of errors.
]],
   noordered = true,
   {name="self", type="tnt.ClassErrorMeter"},
   {name="topk", type="table", default={1}},
   {name="accuracy", type="boolean", default=false},
   call =
      function(self, topk, accuracy)
         self.topk = torch.LongTensor(topk):sort():totable()
         self.accuracy = accuracy
         self:reset()
      end
}

ClassErrorMeter.reset = argcheck{
   {name="self", type="tnt.ClassErrorMeter"},
   call =
      function(self)
         self.sum = {}
         for _,k in ipairs(self.topk) do
            self.sum[k] = 0
         end
         self.n = 0
      end
}

ClassErrorMeter.add = argcheck{
   {name="self", type="tnt.ClassErrorMeter"},
   {name="output", type="torch.*Tensor"},
   {name="target", type="torch.*Tensor"},
   call =
      function(self, output, target)
         target = target:squeeze()
         output = output:squeeze()
         if output:nDimension() == 1 then
            output = output:view(1, output:size(1))
            assert(
               type(target) == 'number',
               'target and output do not match')
            target = torch.Tensor(1):fill(target)
         else
            assert(
               output:nDimension() == 2,
               'wrong output size (1D or 2D expected)')
            assert(
               target:nDimension() == 1,
               'target and output do not match')
         end
         assert(
            target:size(1) == output:size(1),
            'target and output do not match')

         local topk = self.topk
         local maxk = topk[#topk]
         local no = output:size(1)

         local _, pred = output:double():topk(maxk, 2, true, true)
         local correct = pred:typeAs(target):eq(
            target:view(no, 1):expandAs(pred))

         for _,k in ipairs(topk) do
            self.sum[k] = self.sum[k] + no - correct:narrow(2, 1, k):sum()
         end
         self.n = self.n + no
      end
}

ClassErrorMeter.value = argcheck{
   {name="self", type="tnt.ClassErrorMeter"},
   {name="k", type="number", opt=true},
   call =
      function(self, k)
         if k then
            assert(self.sum[k], 'invalid k (this k was not provided at construction time)')
            return self.accuracy and (1-self.sum[k] / self.n)*100 or self.sum[k]*100 / self.n
         else
            local value = {}
            for _,k in ipairs(self.topk) do
               value[k] = self:value(k)
            end
            return value
         end
      end
}
