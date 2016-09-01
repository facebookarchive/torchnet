--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'torch'

local tnt = require 'torchnet.env'
local doc = require 'argcheck.doc'

doc[[
[![Build Status](https://travis-ci.org/torchnet/torchnet.svg)](https://travis-ci.org/torchnet/torchnet)

# torchnet

*torchnet* is a framework for [torch](http://torch.ch) which provides a set
of abstractions aiming at encouraging code re-use as well as encouraging
modular programming.

At the moment, *torchnet* provides four set of important classes:
  - [`Dataset`](#tntdataset): handling and pre-processing data in various ways.
  - [`Engine`](#tntengine): training/testing machine learning algorithm.
  - [`Meter`](#tntmeter): meter performance or any other quantity.
  - [`Log`](#tntlog): output performance or any other string to file / disk in a consistent manner.

For an overview of the *torchnet* framework, please also refer to
[this paper](https://lvdmaaten.github.io/publications/papers/Torchnet_2016.pdf).


## Installation

Please install *torch* first, following instructions on
[torch.ch](http://torch.ch/docs/getting-started.html).  If *torch* is
already installed, make sure you have an up-to-date version of
[*argcheck*](https://github.com/torch/argcheck), otherwise you will get
weird errors at runtime.

Assuming *torch* is already installed, the *torchnet* core is only a set of
lua files, so it is straightforward to install it with *luarocks*
```
luarocks install torchnet
```

To run the MNIST example from the paper, install the `mnist` package:
```
luarocks install mnist
```

`cd` into the installed `torchnet` package directory and run:
```
th example/mnist.lua
```


## Documentation

Requiring *torchnet* returns a local variable containing all *torchnet*
class constructors.
```
local tnt = require 'torchnet'
```

]]

require 'torchnet.dataset'
require 'torchnet.dataset.listdataset'
require 'torchnet.dataset.tabledataset'
require 'torchnet.dataset.indexeddataset'
require 'torchnet.dataset.transformdataset'
require 'torchnet.dataset.batchdataset'
require 'torchnet.dataset.coroutinebatchdataset'
require 'torchnet.dataset.concatdataset'
require 'torchnet.dataset.resampledataset'
require 'torchnet.dataset.shuffledataset'
require 'torchnet.dataset.splitdataset'
require 'torchnet.dataset.datasetiterator'
require 'torchnet.dataset.paralleldatasetiterator'

require 'torchnet.engine'
require 'torchnet.engine.sgdengine'
require 'torchnet.engine.optimengine'

require 'torchnet.meter'
require 'torchnet.meter.apmeter'
require 'torchnet.meter.averagevaluemeter'
require 'torchnet.meter.aucmeter'
require 'torchnet.meter.confusionmeter'
require 'torchnet.meter.mapmeter'
require 'torchnet.meter.msemeter'
require 'torchnet.meter.multilabelconfusionmeter'
require 'torchnet.meter.classerrormeter'
require 'torchnet.meter.timemeter'
require 'torchnet.meter.precisionatkmeter'
require 'torchnet.meter.recallmeter'
require 'torchnet.meter.precisionmeter'
require 'torchnet.meter.ndcgmeter'

require 'torchnet.log'
require 'torchnet.log.remotelog'

require 'torchnet.utils'
require 'torchnet.transform'

require 'torchnet.test.test'

-- function that makes package serializable:
local function _makepackageserializable(packagetbl, packagename)
   local mt = torch.class('package.' .. packagename)
   function mt:__write() end
   function mt:__read()  end
   function mt:__factory() return require(packagename) end
   setmetatable(packagetbl, mt)
end

-- this can be removed when @locronan implements a real torch.isclass():
function torch.isclass(obj)
   local REG = debug.getregistry()
   return REG[obj] and true or false
end

-- make torchnet serializable:
local argcheck = require 'argcheck'
tnt.makepackageserializable = argcheck{
   {name = 'packagetbl',  type = 'table'},
   {name = 'packagename', type = 'string'},
   call = function(packagetbl, packagename)
      assert(not torch.isclass(getmetatable(packagetbl))
         and not torch.isclass(packagetbl), 'input cant be a class (instance)')
      _makepackageserializable(packagetbl, packagename)
      for key, val in pairs(packagetbl) do
         if type(val) == 'table' and not torch.isclass(getmetatable(val))
                                 and not torch.isclass(val) then
            tnt.makepackageserializable(val, packagename .. '.' .. key)
         end
      end
   end
}
tnt.makepackageserializable(tnt, 'torchnet')

return tnt
