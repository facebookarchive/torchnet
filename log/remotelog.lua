--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local threads = require 'threads'
local transfer = require 'torchnet.log.transfer'
local socket = require 'socket'
local argcheck = require 'argcheck'

local RemoteLog, Log = torch.class('tnt.RemoteLog', 'tnt.Log', tnt)

RemoteLog.__clear =
   function(self)
      local server
      if self.__servername then
         server = self.__servername
      else
         self.__mutex = threads.Mutex()
         local servername = torch.ByteStorage()
         servername:retain()
         self.__mutex:lock()

         self.__server = threads.Thread(string.format([[
      local logs = {}
      require 'torch'
      local transfer = require 'torchnet.log.transfer'
      local threads = require 'threads'
      local mutex = threads.Mutex(%d)
      local servername = torch.pushudata(0x%x, "torch.ByteStorage")
      local socket = require("socket")
      local tnt = require 'torchnet'
      -- create a TCP socket and bind it to the local host, at any port
      local server = assert(socket.bind("*", 0))
      -- find out which port the OS chose for us
      local ip, port = server:getsockname()
      servername:string(string.format("%%s:%%s", ip, port))
      mutex:unlock()
      local function error(msg)
          print(string.format("$ Log server %%s:%%d error: %%s", ip, port, msg))
      end
      local function xcall(log, funcname, ...)
        local status, res = pcall(
          function(...)
            return {log[funcname](log, ...)}
          end,
          ...
        )
        if status then
          return table.unpack(res)
        else
          error(res)
        end
      end
      -- loop forever waiting for clients
      while true do
         -- wait for a connection from any client
         local client = server:accept()
         -- make sure we don't block waiting for this client's line
         client:settimeout(10)
         -- receive the line
         local cmd = transfer.receive(client)
         if type(cmd) == 'string' then
            if cmd == 'close' then
               for _, log in pairs(logs) do
                  xcall(log, 'close')
               end
               break
            elseif cmd == 'lognames' then
               local lognames = {}
               for name, _ in pairs(logs) do
                  table.insert(lognames, name)
               end
               transfer.send(client, lognames)
            elseif cmd == 'create' then
               local logname = transfer.receive(client)
               local keys = transfer.receive(client)
               if not logs[logname] then
                  logs[logname] = tnt.Log{keys=keys}
               end
            elseif cmd == 'attach' then
               local logname = transfer.receive(client)
               local event = transfer.receive(client)
               local closures = transfer.receive(client)
               xcall(logs[logname], 'attach', event, closures)
            elseif cmd == 'set' then
               local logname = transfer.receive(client)
               local keys = transfer.receive(client)
               xcall(logs[logname], 'set', keys)
            elseif cmd == 'get' then
               local logname = transfer.receive(client)
               local key = transfer.receive(client)
               local value = xcall(logs[logname], 'get', key)
               transfer.send(client, value)
            elseif cmd == 'flush' then
               local logname = transfer.receive(client)
               xcall(logs[logname], 'flush')
            end
         end
         client:close()
      end
      server:close()
]], self.__mutex:id(), torch.pointer(servername)))

         self.__mutex:lock()
         server = servername:string()

         -- GC Lua 5.1
         if newproxy then
            self.__gc__ = newproxy(true)
            getmetatable(self.__gc__).__gc =
               function()
                  self:__gc()
               end
         end
      end

      self.__ip, self.__port = server:match("^(.+)%:(.+)$")
      self.__port = tonumber(self.__port)
      assert(self.__ip and self.__port, "invalid ip:port name")

      -- create table
      local c = socket.connect(self.__ip, self.__port)
      transfer.send(c, "create")
      transfer.send(c, self.__name)
      local keys = {}
      for key, _ in pairs(self.__keys) do
         table.insert(keys, key)
      end
      transfer.send(c, keys)
      c:close()
   end

RemoteLog.__init = argcheck{
   doc = [[
<a name="RemoteLog">
#### tnt.RemoteLog(@ARGP)
@ARGT

Creates a new `RemoteLog` with allowed keys (strings) `keys`.  Specifiy event
closures with table of functions `onClose`, `onFlush`, `onGet` and `onSet`,
which will be called when `close()`, `flush()`, `get()`, and `set{}`
methods will be called, respectively.

If `server` is not provided, `RemoteLog` creates a server which can later be
reached at the address provided by `server()`.

If `server` is provided, `RemoteLog` will dialog with the given server to
store any values to be recorded by the `Log` (or query any of these values).

A given server can record different `Log`, with different names. The default name
is `default`, but can be specified with the `name` option.

At this time, it is important to call the `close()` method when `RemoteLog`
is not used anymore (before quitting the application).
]],
   noordered=true,
   {name="self", type="tnt.RemoteLog"},
   {name="keys", type="table"},
   {name="server", type="string", opt=true},
   {name="name", type="string", default="default"},
   {name="onClose", type="table", opt=true},
   {name="onFlush", type="table", opt=true},
   {name="onGet", type="table", opt=true},
   {name="onSet", type="table", opt=true},
   call =
      function(self, keys, server, name, onClose, onFlush, onGet, onSet)
         self.__server = server
         self.__name = name
         Log.__init(
            self,
            {
               keys=keys,
               onClose=onClose,
               onFlush=onFlush,
               onGet=onGet,
               onSet=onSet
            }
         )
      end
}

RemoteLog.set = argcheck{
   nonamed=true,
   {name="self", type="tnt.RemoteLog"},
   {name="keys", type="table"},
   call =
      function(self, keys)
         local c = socket.connect(self.__ip, self.__port)
         transfer.send(c, "set")
         transfer.send(c, self.__name)
         transfer.send(c, keys)
         c:close()
      end
}

RemoteLog.get = argcheck{
   {name="self", type="tnt.Log"},
   {name="key", type="string"},
   call =
      function(self, key)
         local c = socket.connect(self.__ip, self.__port)
         transfer.send(c, "get")
         transfer.send(c, self.__name)
         transfer.send(c, key)
         local value = transfer.receive(c)
         c:close()
         return value
      end
}

RemoteLog.flush = argcheck{
   {name="self", type="tnt.RemoteLog"},
   call =
      function(self)
         local c = socket.connect(self.__ip, self.__port)
         transfer.send(c, "flush")
         transfer.send(c, self.__name)
         c:close()
      end
}

RemoteLog.attach = argcheck{
   {name="self", type="tnt.RemoteLog"},
   {name="event", type="string"},
   {name="closures", type="table"},
   call =
      function(self, event, closures)
         local c = socket.connect(self.__ip, self.__port)
         transfer.send(c, "attach")
         transfer.send(c, self.__name)
         transfer.send(c, event)
         transfer.send(c, closures)
         c:close()
      end
}

RemoteLog.close = argcheck{
   {name="self", type="tnt.RemoteLog"},
   call =
      function(self)
         local c = socket.connect(self.__ip, self.__port)
         transfer.send(c, "close")
         c:close()
      end
}

RemoteLog.server = argcheck{
   {name="self", type="tnt.RemoteLog"},
   call =
      function(self)
         return string.format("%s:%s", self.__ip, self.__port)
      end
}

RemoteLog.lognames = argcheck{
   {name="self", type="tnt.RemoteLog"},
   call =
      function(self)
         local c = socket.connect(self.__ip, self.__port)
         transfer.send(c, "lognames")
         local lognames = transfer.receive(c)
         c:close()
         return lognames
      end
}

-- GC Lua 5.2
function RemoteLog:__gc()
   if self.__server then
      local c = socket.connect(self.__ip, self.__port)
      transfer.send(c, "close")
      c:close()
      self.__server:free()
   end
end
