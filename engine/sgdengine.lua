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

### tnt.SGDEngine

The `SGDEngine` module implements the Stochastic Gradient Descent training
procedure in `train`, including data sampling, forward prop, back prop, and
parameter updates. It also operates as a coroutine allowing a user control
 (i.e. increment some sort of `tnt.Meter`) at events such as 'start',
'start-epoch', 'forward', 'forward-criterion', 'backward', etc.
The available hooks are the following:
```lua
hooks = {
   ['onStart']             = function() end, -- Right before training
   ['onStartEpoch']        = function() end, -- Before new epoch
   ['onSample']            = function() end, -- After getting a sample
   ['onForward']           = function() end, -- After model:forward
   ['onForwardCriterion']  = function() end, -- After criterion:forward
   ['onBackwardCriterion'] = function() end, -- After criterion:backward
   ['onBackward']          = function() end, -- After model:backward
   ['onUpdate']            = function() end, -- After UpdateParameters
   ['onEndEpoch']          = function() end, -- Right before completing epoch
   ['onEnd']               = function() end, -- After training
}
```
To specify a new closure for a given hook, we can access to it with
`engine.hooks.<onEvent>`. For example, we could reset a `Meter` before every
epoch by:
```lua
local engine = tnt.SGDEngine()
local meter  = tnt.AverageValueMeter()
engine.hooks.onStartEpoch = function(state)
   meter:reset()
end
```

Accordingly, `train` requires a network (`nn.Module`), a criterion expressing the
loss function (`nn.Criterion`), a dataset iterator (`tnt.DatasetIterator`), and a
learning rate, at the minimum. The `test` function allows for simple evaluation
of a model on a dataset.

A `state` is maintained for external access to outputs and parameters of modules
as well as sampled data. The content of the `state` table is the following, where
the passed values come from the arguments of `engine:train()`:
```lua
state = {
   ['network']     = network,
   ['criterion']   = criterion,
   ['iterator']    = iterator,
   ['lr']          = lr,
   ['lrcriterion'] = lrcriterion,
   ['maxepoch']    = maxepoch,
   ['sample']      = {},
   ['epoch']       = 0, -- epoch done so far
   ['t']           = 0, -- samples seen so far
   ['training']    = true
}
```
]]

require 'nn'

local SGDEngine, Engine = torch.class('tnt.SGDEngine', 'tnt.Engine', tnt)

SGDEngine.__init = argcheck{
   {name="self", type="tnt.SGDEngine"},
   call =
      function(self)
         Engine.__init(self, {
            "onStart", "onStartEpoch", "onSample",
            "onForward", "onForwardCriterion",
            "onBackward", "onBackwardCriterion",
            "onEndEpoch", "onUpdate", "onEnd"
         })
      end
}

SGDEngine.train = argcheck{
   {name="self", type="tnt.SGDEngine"},
   {name="network", type="nn.Module"},
   {name="criterion", type="nn.Criterion"},
   {name="iterator", type="tnt.DatasetIterator"},
   {name="lr", type="number"},
   {name="lrcriterion", type="number", defaulta="lr"},
   {name="maxepoch", type="number", default=1000},
   call =
      function(self, network, criterion, iterator, lr, lrcriterion, maxepoch)
         local state = {
            network = network,
            criterion = criterion,
            iterator = iterator,
            lr = lr,
            lrcriterion = lrcriterion,
            maxepoch = maxepoch,
            sample = {},
            epoch = 0, -- epoch done so far
            t = 0, -- samples seen so far
            training = true
         }

         self.hooks("onStart", state)
         while state.epoch < state.maxepoch do
            state.network:training()

            self.hooks("onStartEpoch", state)
            for sample in state.iterator() do
               state.sample = sample
               self.hooks("onSample", state)

               state.network:forward(sample.input)
               self.hooks("onForward", state)
               state.criterion:forward(state.network.output, sample.target)
               self.hooks("onForwardCriterion", state)

               state.network:zeroGradParameters()
               if state.criterion.zeroGradParameters then
                  state.criterion:zeroGradParameters()
               end

               state.criterion:backward(state.network.output, sample.target)
               self.hooks("onBackwardCriterion", state)
               state.network:backward(sample.input, state.criterion.gradInput)
               self.hooks("onBackward", state)

               assert(state.lrcriterion >= 0, 'lrcriterion should be positive or zero')
               if state.lrcriterion > 0 and state.criterion.updateParameters then
                  state.criterion:updateParameters(state.lrcriterion)
               end
               assert(state.lr >= 0, 'lr should be positive or zero')
               if state.lr > 0 then
                  state.network:updateParameters(state.lr)
               end
               state.t = state.t + 1
               self.hooks("onUpdate", state)
            end
            state.epoch = state.epoch + 1
            self.hooks("onEndEpoch", state)
         end
         self.hooks("onEnd", state)
      end
}

SGDEngine.test = argcheck{
   {name="self", type="tnt.SGDEngine"},
   {name="network", type="nn.Module"},
   {name="iterator", type="tnt.DatasetIterator"},
   {name="criterion", type="nn.Criterion", opt=true},
   call = function(self, network, iterator, criterion)
      local state = {
         network = network,
         iterator = iterator,
         criterion = criterion,
         sample = {},
         t = 0, -- samples seen so far
         training = false
      }

      self.hooks("onStart", state)
      state.network:evaluate()
      for sample in state.iterator() do
         state.sample = sample
         self.hooks("onSample", state)
         state.network:forward(sample.input)
         state.t = state.t + 1
         self.hooks("onForward", state)

         if state.criterion then
            state.criterion:forward(state.network.output, sample.target)
            self.hooks("onForwardCriterion", state)
         end

      end
      self.hooks("onEnd", state)
   end
}
