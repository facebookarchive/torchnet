--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local doc = require 'argcheck.doc'

doc[[

### tnt.utils

*Torchnet* provides a set of util functions which are used all over torchnet.
]]

local utils = {}
tnt.utils = utils

utils.table = require 'torchnet.utils.table'
utils.nn = require 'torchnet.utils.nn'
utils.sys = require 'torchnet.utils.sys'

return utils
