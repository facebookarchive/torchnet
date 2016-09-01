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

local Log = torch.class('tnt.Log', tnt)

doc[[
### tnt.Log

`Log` classes act as tables indexed by string keys. Allowed keys must be
provided at construction. A special key `__status__` can be also set the
convenience method `log:status()` to record basic messages.

Viewers closures can be attached to a `Log`, and called at different events:
   * `onSet(log, key, value)`: when setting a key to the `Log` with `log:set{}`.
   * `onGet(log, key)`: when querying a key with `log:get()`.
   * `onFlush(log)`: when flushing out the stored data of the `Log` with `log:flush()`.
   * `onClose(log)`: when closing a `Log` with `log:close()`.

Typical viewer closures are `text` or `json`, which allow to write to disk
or to the console a subset of the keys stored by the `Log`, in a particular
format. The special viewer closure `status` is made to be called on `set()`
events, and will print out only status records.

A typical use case would be the following:
```lua
tnt = require 'torchnet'

-- require the viewers we want
logtext = require 'torchnet.log.view.text'
logstatus = require 'torchnet.log.view.status'

log = tnt.Log{
   keys = {"loss", "accuracy"},
   onFlush = {
      -- write out all keys in "log" file
      logtext{filename='log.txt', keys={"loss", "accuracy"}, format={"%10.5f", "%3.2f"}},
      -- write out loss in a standalone file
      logtext{filename='loss.txt', keys={"loss"}},
      -- print on screen too
      logtext{keys={"loss", "accuracy"}},
   },
   onSet = {
      -- add status to log
      logstatus{filename='log.txt'},
      -- print status to screen
      logstatus{},
   }
}

-- set values
log:set{
  loss = 0.1,
  accuracy = 97
}

-- write some info
log:status("hello world")

-- flush out log
log:flush()
```
]]

Log.__clear =
   function(self)
      self.__events = {onClose={}, onFlush={}, onGet={}, onSet={}}
      self.__data = {}
   end

Log.__init = argcheck{
   doc = [[
<a name="Log">
#### tnt.Log(@ARGP)
@ARGT

Creates a new `Log` with allowed keys (strings) `keys`.  Specifiy event
closures with table of functions `onClose`, `onFlush`, `onGet` and `onSet`,
which will be called when `close()`, `flush()`, `get()`, and `set{}`
methods will be called, respectively.
]],
   noordered=true,
   {name="self", type="tnt.Log"},
   {name="keys", type="table"},
   {name="onClose", type="table", opt=true},
   {name="onFlush", type="table", opt=true},
   {name="onGet", type="table", opt=true},
   {name="onSet", type="table", opt=true},
   call =
      function(self, keys, onClose, onFlush, onGet, onSet)
         self.__keys = {__status__ = true}
         for _, key in ipairs(keys) do
            self.__keys[key] = true
         end
         self:__clear()
         if onClose then
            self:attach('onClose', onClose)
         end
         if onFlush then
            self:attach('onFlush', onFlush)
         end
         if onGet then
            self:attach('onGet', onGet)
         end
         if onSet then
            self:attach('onSet', onSet)
         end
     end
}

Log.status = argcheck{
   doc = [[
<a name="Log.status">
#### tnt.Log:status(@ARGP)
@ARGT

Record a status message, with corresponding (optional) time of the event.
]],
   {name="self", type="tnt.Log"},
   {name="message", type="string", opt=true},
   {name="time", type="boolean", default=true},
   call =
      function(self, message, time)
         local prefix = "|"
         if time then
            prefix = prefix .. " " .. os.date() .. " |"
         end
         self:set{
            __status__ = string.format("%s %s", prefix, message)
         }
      end
}

Log.set = argcheck{
   doc = [[
<a name="Log.set">
#### tnt.Log:set(@ARGP)
@ARGT

Set a number of keys (a subset of the keys provided at construction) to
their corresponding values.

Closures attached to the `onSet(log, key, value)` event will be called.
]],
   nonamed=true,
   {name="self", type="tnt.Log"},
   {name="keys", type="table"},
   call =
      function(self, keys)
         for key, value in pairs(keys) do
            assert(type(key) == 'string', 'string expected for key')
            if not self.__keys[key] then
               error(string.format("unknown key <%s>", key))
            end
            for _, closure in ipairs(self.__events.onSet) do
               closure(self, key, value)
            end
            self.__data[key] = value
         end
      end
}

Log.get = argcheck{
   doc = [[
<a name="Log.get">
#### tnt.Log:get(@ARGP)
@ARGT

Get the value of a given key.

Closures attached to the `onGet(log, key)` event will be called.
]],
   {name="self", type="tnt.Log"},
   {name="key", type="string"},
   call =
      function(self, key)
         if not self.__keys[key] then
            error(string.format("unknown key <%s>", key))
         end
         for _, closure in ipairs(self.__events.onGet) do
            closure(self, key)
         end
         return self.__data[key]
      end
}

Log.flush = argcheck{
   doc = [[
<a name="Log.flush">
#### tnt.Log:flush(@ARGP)
@ARGT

Flush (empty) the log data.

Closures attached to the `onFlush(log)` event will be called.
]],
   {name="self", type="tnt.Log"},
   call =
      function(self)
         for _, closure in ipairs(self.__events.onFlush) do
            closure(self)
         end
         self.__data = {}
      end
}

Log.close = argcheck{
   doc = [[
<a name="Log.close">
#### tnt.Log:close(@ARGP)
@ARGT

Close the log.

Closures attached to the `onClose(log)` event will be called.
]],
   {name="self", type="tnt.Log"},
   call =
      function(self)
         for _, closure in ipairs(self.__events.onClose) do
            closure(self)
         end
         self:__clear()
      end
}

Log.attach = argcheck{
   doc = [[
<a name="Log.attach">
#### tnt.Log:attach(@ARGP)
@ARGT

Attach a set of functions (provided in a table) to a given event.
]],
   {name="self", type="tnt.Log"},
   {name="event", type="string"},
   {name="closures", type="table"},
   call =
      function(self, event, closures)
         local events = self.__events[event]
         assert(events, string.format('unknown event <%s>', event))
         for _, closure in ipairs(closures) do
            assert(type(closure) == 'function', string.format('%s: table of functions expected', event))
            table.insert(events, closure)
         end
      end
}
