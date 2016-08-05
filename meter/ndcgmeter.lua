--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local NDCGMeter = torch.class('tnt.NDCGMeter', 'tnt.Meter', tnt)

-- function computing discounted cumulative gain:
local function computeDCG(relevance, index, K)

   -- assertions:
   assert(relevance)
   assert(index)
   assert(K)
   assert(type(K) == 'number')
   assert(index:max() <= relevance:nElement())
   assert(index:nElement() >= K)
   relevance = relevance:squeeze()
   index     = index:squeeze()
   assert(relevance:dim() == 1)
   assert(index:dim() == 1)

   -- return DCG:
   local dcg = relevance[index[1]]
   if K > 1 then
      dcg = dcg + relevance:index(1, index:narrow(1, 2, K - 1)):cdiv(
         torch.range(2, K):log():div(math.log(2)):typeAs(relevance)
      ):sum()
   end
   return dcg
end

-- function computing ideal discounted cumulative gain:
local function computeIDCG(relevance, K)
   relevance = relevance:squeeze()
   assert(relevance:dim() == 1)
   local _,sortind = torch.sort(relevance, 1, true)  -- descending order
   return computeDCG(relevance, sortind, K)
end

-- function computing the normalized discounted cumulative gain:
local function computeNCDG(relevance, index, K)
   local r = computeDCG(relevance, index, K) / computeIDCG(relevance, K)
   assert(r >= 0 and r <= 1)
   return r
end

NDCGMeter.__init = argcheck{
   doc = [[
<a name="NDCGMeter">
#### tnt.NDCGMeter(@ARGP)
@ARGT

The `tnt.NDCGMeter` measures the normalized discounted cumulative gain (NDCG) of
a ranking produced by a model at prespecified levels k, and averages the NDCG
over all examples.

The discounted cumulative gain at level k is defined as:

DCG_k = rel_1 + \sum{i = 2}^k (rel_i / log_2(i))

Herein, rel_i is the relevance of item i as specified by an external rater.
Defining ideal DCG (IDCG) as the best possible DCG for a given example, the NDCG
at level k is defined as:

NDCG_k = DCG_k / IDCG_k

At initialization time, the meter takes as input a table `K` that contains all
the levels k at which the NDCG is computed.

The `add(output, relevance)` method takes as input (1) a NxC tensor of model
`outputs`, which scores for all C possible outputs for a batch of N examples;
and (2) a NxC tensor `relevance` that contains the corresponding relevances for
these scores, as provided by an external rater. Relevances are generally
obtained from human raters.

The `value()` method returns a table that contains the NDCG values for all
levels K that were provided at initialization time. Alternatively, the NDCG at
a specific level k can be obtained by calling `value(k)`. Note that the level
`k` should be an element of the table `K` specified at initialization time.

Please note that the number of outputs and relevances C should always be at
least as high as the highest NDCG level k that the meter is computing.
]],
   {name="self", type="tnt.NDCGMeter"},
   {name="K",    type="table", default = {1}},
   noordered=true,
   call =
      function(self, K)
         self.K = torch.LongTensor(K):sort():totable()
         self:reset()
      end
}

NDCGMeter.reset = argcheck{
   {name="self", type="tnt.NDCGMeter"},
   call =
      function(self)
         self.ndcg = {}
         for _,k in ipairs(self.K) do self.ndcg[k] = 0 end
         self.n = 0
      end
}

NDCGMeter.add = argcheck{
   {name="self", type="tnt.NDCGMeter"},
   {name="output",    type="torch.*Tensor"},
   {name="relevance", type="torch.*Tensor"},
   call =
      function(self, output, relevance)

         -- check inputs:
         if output:dim() == 1 then
            output:resize(1, output:nElement())
         end
         if relevance:dim() == 1 then
            relevance:resize(1, relevance:nElement())
         end
         assert(output:dim() == 2)
         assert(relevance:dim() == 2)
         assert(output:size(1) == relevance:size(1), 'batch size must match')
         assert(output:size(2) == relevance:size(2), 'result size must match')
         assert(
            relevance:size(2) >= self.K[#self.K],
            'too few results for value of K'
         )

         -- compute average NDCG:
         relevance = relevance:double()
         local _,index = torch.sort(output, 2, true)  -- descending order
         for n = 1,index:size(1) do
            for _,k in ipairs(self.K) do
               self.ndcg[k] =
                  self.ndcg[k] + computeNCDG(relevance[n], index[n], k)
            end
         end
         self.n = self.n + index:size(1)
      end
}

NDCGMeter.value = argcheck{
   {name="self", type="tnt.NDCGMeter"},
   {name="K", type="number", opt=true},
   call =
      function(self, K)
         if K then
            assert(
               self.ndcg[K], 'invalid k (was not provided at construction time)'
            )
            return self.ndcg[K] / self.n
         else
            local value = {}
            for _,k in ipairs(self.K) do
               value[k] = self.ndcg[k] / self.n
            end
            return value
         end
      end
}
