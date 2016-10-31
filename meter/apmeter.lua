--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local APMeter = torch.class('tnt.APMeter', 'tnt.Meter', tnt)


APMeter.__init = argcheck{
   doc = [[
<a name="APMeter">
#### tnt.APMeter(@ARGP)
@ARGT

The `tnt.APMeter` measures the average precision per class.

The `tnt.APMeter` is designed to operate on `NxK` Tensors `output` and
`target`, and optionally a `Nx1` Tensor weight where (1) the `output` contains
model output scores for `N` examples and `K` classes that ought to be higher
when the model is more convinced that the example should be positively labeled,
and smaller when the model believes the example should be negatively labeled
(for instance, the output of a sigmoid function); (2) the `target` contains
only values 0 (for negative examples) and 1 (for positive examples); and (3)
the `weight` ( > 0) reprsents weight for each sample.

The `tnt.APMeter` has no parameters to be set.
]],
   {name="self", type="tnt.APMeter"},
   call = function(self)
      self:reset()
   end
}

APMeter.reset = argcheck{
   {name="self", type="tnt.APMeter"},
   call = function(self)
      self.scores  = torch.DoubleTensor(torch.DoubleStorage())
      self.targets = torch.LongTensor(  torch.LongStorage())
      self.weights = torch.DoubleTensor(torch.DoubleStorage())
   end
}

APMeter.add = argcheck{
   {name="self", type="tnt.APMeter"},
   {name="output", type="torch.*Tensor"},
   {name="target", type="torch.*Tensor"},
   {name="weight", type="torch.*Tensor", opt=true},
   call = function(self, output, target, weight)

      -- assertions on the input:
      if weight then
         weight = weight:squeeze()
      end
      if output:nDimension() == 1 then
         output = output:view(output:size(1), 1)
      else
         assert(output:nDimension() == 2,
            'wrong output size (should be 1D or 2D with one column per class)'
         )
      end
      if target:nDimension() == 1 then
         target = target:view(target:size(1), 1)
      else
         assert(target:nDimension() == 2,
            'wrong target size (should be 1D or 2D with one column per class)'
         )
      end
      if weight then
        assert(weight:nDimension() == 1, 'Weight dimension should be 1')
        assert(weight:nElement() == target:size(1),
          'Weight dimension 1 should be the same as that of target'
        )
        assert(torch.ge(weight, 0):all(), 'Weight should be non-negative only')
      end

      assert(output:size(1) == target:size(1) and
             output:size(2) == target:size(2),
         'dimensions for output and target does not match'
      )
      assert(torch.eq(torch.eq(target, 0):add(torch.eq(target, 1)), 1):all(),
         'targets should be binary (0 or 1)'
      )
      if self.scores:nElement() > 0 then
         assert(output:size(2) == self.scores:size(2),
            'dimensions for output should match previously added examples.'
         )
      end
      if self.targets:nElement() > 0 then
         assert(target:size(2) == self.targets:size(2),
            'dimensions for output should match previously added examples.'
         )
      end

      -- make sure storage is of sufficient size:
      if self.scores:storage():size() < self.scores:nElement() + output:nElement() then
         local newsize = math.ceil(self.scores:storage():size() * 1.5)
         local newweightsize = math.ceil(self.weights:storage():size() * 1.5)
         self.scores:storage():resize(newsize + output:nElement())
         self.targets:storage():resize(newsize + output:nElement())
         if weight then
           self.weights:storage():resize(newweightsize + output:size(1))
         end
      end

      -- store scores and targets:
      local offset = (self.scores:dim() > 0) and self.scores:size(1) or 0
      self.scores:resize(offset + output:size(1), output:size(2))
      self.targets:resize(offset + target:size(1), target:size(2))

      self.scores:narrow(1, offset + 1, output:size(1)):copy(output)
      self.targets:narrow(1, offset + 1, target:size(1)):copy(target)

      if weight then
        self.weights:resize(offset + weight:size(1))
        self.weights:narrow(1, offset + 1, weight:size(1)):copy(weight)
      end
   end
}

APMeter.value = argcheck{
   {name="self", type="tnt.APMeter"},
   call = function(self)

      -- compute average precision for each class:
      if not self.scores:nElement() == 0 then return 0 end
      local ap = torch.DoubleTensor(self.scores:size(2)):fill(0)
      local range = torch.range(1, self.scores:size(1), 'torch.DoubleTensor')
      local weight, weightedtruth
      if self.weights:nElement() > 0 then
        weight = self.weights.new(self.weights:size())
        weightedtruth = self.weights.new(self.weights:size())
      end
      for k = 1,self.scores:size(2) do

         -- sort scores:
         local scores  =  self.scores:select(2, k)
         local targets = self.targets:select(2, k)
         local _,sortind = torch.sort(scores, 1, true)
         local truth = targets:index(1, sortind)
         if self.weights:nElement() > 0 then
            weight:index(self.weights, 1, sortind)
            torch.cmul(weightedtruth, truth:double(), weight)
            range = weight:cumsum()
         end
         -- compute true positive sums:
         local tp = weightedtruth and weightedtruth:cumsum()
                      or truth:double():cumsum()

         -- compute precision curve:
         local precision = tp:cdiv(range)

         -- compute average precision:
         ap[k] = precision[truth:byte()]:sum() / math.max(truth:sum(), 1)
      end
      return ap
   end
}
